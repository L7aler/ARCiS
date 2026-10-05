#!/usr/bin/env python3
"""Run an ARCiS grid and replace a PyARCiS worker after GGchem fails.

Each worker keeps its initialized opacity tables while it is healthy.  When
GGchem creates ``fatal.case``, the supervisor immediately terminates that
worker, records the failed attempt, and starts a fresh process in the same
logical slot.  The fresh worker retries that case with pmax divided by ten.
Each worker has a private result pipe, so terminating one worker cannot block
messages from any of the healthy workers.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="arcis_noclouds_grid_lumen.dat")
    parser.add_argument("--input", default="output_gas_test/arcis_gas.dat")
    parser.add_argument("--output-root", default="output_gas_test")
    parser.add_argument("--work-root", default="arcis_work")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--fatal-retries",
        type=int,
        default=2,
        help="Number of fresh-process retries after the first GGchem failure.",
    )
    parser.add_argument(
        "--base-pmax",
        type=float,
        default=None,
        help=(
            "Initial pmax in bar. By default it is read from the input file; "
            "a pmax column in the grid overrides it for individual cases."
        ),
    )
    parser.add_argument("--planet-mass", type=float, default=1.0)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--home",
        default="/net/lumen/data2/louis",
        help="HOME used by ARCiS to locate its Data directory.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run rows already marked converged as well.",
    )
    return parser.parse_args()


def read_grid(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return rows, list(reader.fieldnames)


def read_input_float(path: Path, keyword: str) -> float:
    """Read a scalar numeric keyword from an ARCiS input file."""
    target = keyword.strip().lower()
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().lower() != target:
                continue
            return parse_float(value.strip().split()[0])
    raise ValueError(f"Keyword {keyword!r} was not found in {path}")


def parse_float(value: Any) -> float:
    """Parse ordinary and Fortran-style floating-point values."""
    return float(str(value).strip().replace("D", "E").replace("d", "e"))


def write_grid_atomic(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_temperature_convergence(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open() as handle:
        for line in handle:
            key, _, value = line.partition("=")
            if key.strip().lower() == "converged":
                return value.strip().lower() == "true"
    return False


def define_radius(planet_mass_jupiter: float, logg_cgs: float) -> float:
    """Return Rp/Rjup for Mp/Mjup and log10(g [cm s^-2])."""
    #gravitational_constant = 6.67430e-8
    #jupiter_mass = 1.89816e30
    #jupiter_radius = 7.1492e9
    #Actual values used in ARCiS internally
    gravitational_constant = 6.6740831e-8
    jupiter_mass  = 1.898e30
    jupiter_radius  = 6.9911e9
    gravity = 10.0**logg_cgs
    return (
        math.sqrt(
            gravitational_constant * planet_mass_jupiter * jupiter_mass / gravity
        )
        / jupiter_radius
    )


def clear_output(output_dir: Path) -> None:
    """Clear generated output but retain the cumulative ARCiS log."""
    for item in output_dir.iterdir():
        if item.name == "log.dat":
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def case_directory(output_root: Path, case: dict[str, str]) -> Path:
    return output_root / (
        f"case_{case['case_id']}"
        f"_Dplanet_{case['Dplanet']}"
        f"_Tint_{case['Tint']}"
        f"_logg_{case['logg']}"
        f"_met_{case['[M/H]']}"
    )


def copy_case_output(
    output_dir: Path,
    output_root: Path,
    case: dict[str, str],
    retry: int,
) -> Path:
    destination = case_directory(output_root, case)
    if destination.exists():
        destination = destination.with_name(
            destination.name + f"_rerun_{retry}_{time.time_ns()}"
        )
    shutil.copytree(output_dir, destination)
    return destination


def send_and_exit(result_connection: Any, message: tuple[Any, ...]) -> None:
    """Synchronously report a fatal/error message and close this worker's pipe."""
    result_connection.send(message)
    result_connection.close()


def worker_main(
    token: str,
    task_queue: mp.Queue,
    result_connection: Any,
    input_file_string: str,
    output_root_string: str,
    work_root_string: str,
    planet_mass: float,
    default_pmax: float,
) -> None:
    # Import the extension only inside a freshly spawned process.
    import pyARCiS

    output_root = Path(output_root_string)
    worker_dir = Path(work_root_string) / token
    output_dir = output_root / "worker_output" / token
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(worker_dir)

    output_dir_string = str(output_dir) + "/"
    try:
        pyARCiS.pyinit(input_file_string, output_dir_string)
        pyARCiS.pyverbose(True)
    except BaseException:
        send_and_exit(
            result_connection,
            ("init_error", token, traceback.format_exc()),
        )
        return

    result_connection.send(("ready", token))

    while True:
        task = task_queue.get()
        if task is None:
            result_connection.close()
            return

        case, retry = task
        case_id = str(case["case_id"])
        case_pmax = case.get("pmax", "")
        base_pmax = parse_float(case_pmax) if str(case_pmax).strip() else default_pmax
        pmax_used = base_pmax / (10.0**retry)
        result_connection.send(("started", token, case_id, retry))
        fatal_file = worker_dir / "fatal.case"
        fatal_file.unlink(missing_ok=True)

        try:
            print(
                f"[{token}] case={case_id}, retry={retry}, "
                f"Dplanet={case['Dplanet']}, [M/H]={case['[M/H]']}, "
                f"Tint={case['Tint']}, logg={case['logg']}, "
                f"pmax={pmax_used:g} bar",
                flush=True,
            )

            pyARCiS.pysetvalue("Mp", float(planet_mass))
            pyARCiS.pysetvalue("pmax", pmax_used)
            pyARCiS.pysetvalue("Dplanet", float(case["Dplanet"]))
            pyARCiS.pysetvalue("TeffP", float(case["Tint"]))
            pyARCiS.pysetvalue("metallicity", float(case["[M/H]"]))
            radius = define_radius(planet_mass, float(case["logg"]))
            pyARCiS.pysetvalue("Rp", float(radius))

            pyARCiS.pycomputemodel()

            if fatal_file.exists():
                failure_dir = (
                    output_root
                    / "fatal_cases"
                    / f"case_{case_id}_retry_{retry}_{token}"
                )
                failure_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fatal_file, failure_dir / "fatal.case")
                (failure_dir / "parameters.txt").write_text(
                    "\n".join(f"{key}={value}" for key, value in case.items())
                    + f"\nfatal_retry={retry}\n"
                    + f"base_pmax={base_pmax:.16g}\n"
                    + f"pmax_used={pmax_used:.16g}\n"
                )
                send_and_exit(
                    result_connection,
                    (
                        "fatal",
                        token,
                        case_id,
                        retry,
                        fatal_file.read_text(errors="replace"),
                    ),
                )
                return

            pyARCiS.pywritefiles()
            destination = copy_case_output(
                output_dir,
                output_root,
                case,
                retry,
            )
            converged = read_temperature_convergence(
                destination / "temperature_convergence.dat"
            )
            (destination / "run_metadata.txt").write_text(
                f"fatal_retry={retry}\n"
                f"base_pmax={base_pmax:.16g}\n"
                f"pmax_used={pmax_used:.16g}\n"
            )
            clear_output(output_dir)
            result_connection.send(
                ("done", token, case_id, retry, converged, str(destination))
            )
            result_connection.send(("ready", token))

        except BaseException:
            send_and_exit(
                result_connection,
                ("error", token, case_id, retry, traceback.format_exc()),
            )
            return


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.fatal_retries < 0:
        raise ValueError("--fatal-retries cannot be negative")
    if args.base_pmax is not None and args.base_pmax <= 0:
        raise ValueError("--base-pmax must be greater than zero")
    # These values are inherited by spawned workers before PyARCiS is loaded.
    os.environ["HOME"] = args.home
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")

    launch_dir = Path.cwd().resolve()
    grid_file = (launch_dir / args.grid).resolve()
    input_file = (launch_dir / args.input).resolve()
    output_root = (launch_dir / args.output_root).resolve()
    run_id = f"run_{os.getpid()}_{time.time_ns()}"
    work_root = (launch_dir / args.work_root / run_id).resolve()

    if not grid_file.is_file():
        raise FileNotFoundError(grid_file)
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    default_pmax = (
        args.base_pmax
        if args.base_pmax is not None
        else read_input_float(input_file, "pmax")
    )
    if default_pmax <= 0:
        raise ValueError("The initial pmax must be greater than zero")
    print(f"Default initial pmax: {default_pmax:g} bar", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_grid(grid_file)
    required = {"case_id", "Dplanet", "[M/H]", "Tint", "logg"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Grid is missing columns: {sorted(missing)}")
    for optional in ("attempt", "converged"):
        if optional not in fieldnames:
            fieldnames.append(optional)
            for row in rows:
                row[optional] = "0"

    selected = [
        row
        for row in rows
        if args.run_all or not is_true(row.get("converged", "0"))
    ]
    if not selected:
        print("No unconverged cases remain.")
        return 0

    row_by_id = {str(row["case_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("case_id values must be unique")

    pending: deque[tuple[dict[str, str], int]] = deque(
        (case, 0) for case in selected
    )
    fresh_retry_by_slot: dict[int, tuple[dict[str, str], int]] = {}
    completed: set[str] = set()
    permanently_failed: set[str] = set()

    context = mp.get_context("spawn")
    workers: dict[str, dict[str, Any]] = {}
    ready: set[str] = set()
    assigned: dict[str, tuple[dict[str, str], int]] = {}
    inflight: dict[str, tuple[dict[str, str], int]] = {}
    forced_fatal: dict[str, tuple[dict[str, str], int]] = {}
    generations = [0 for _ in range(args.workers)]

    def start_worker(slot: int) -> None:
        generation = generations[slot]
        generations[slot] += 1
        token = f"worker_{slot:03d}_generation_{generation:04d}"
        private_queue = context.Queue(maxsize=1)
        result_receiver, result_sender = context.Pipe(duplex=False)
        process = context.Process(
            target=worker_main,
            args=(
                token,
                private_queue,
                result_sender,
                str(input_file),
                str(output_root),
                str(work_root),
                args.planet_mass,
                default_pmax,
            ),
            name=token,
        )
        process.start()
        # Only the worker writes to this end. Closing the parent's duplicate is
        # important so EOF is observable when the worker exits.
        result_sender.close()
        workers[token] = {
            "process": process,
            "queue": private_queue,
            "result": result_receiver,
            "slot": slot,
        }
        print(f"Started {token} (PID {process.pid})", flush=True)

    def checkpoint(case_id: str, converged: bool) -> None:
        row = row_by_id[case_id]
        row["attempt"] = "1"
        row["converged"] = "1" if converged else "0"
        write_grid_atomic(grid_file, rows, fieldnames)

    def mark_failed(
        task: tuple[dict[str, str], int],
        reason: str,
    ) -> None:
        case, retry = task
        case_id = str(case["case_id"])
        permanently_failed.add(case_id)
        checkpoint(case_id, False)
        print(
            f"Case {case_id} failed and will not be retried: {reason}",
            flush=True,
        )

    def retry_or_fail(
        slot: int,
        task: tuple[dict[str, str], int],
        reason: str,
    ) -> None:
        case, retry = task
        case_id = str(case["case_id"])
        case_pmax = case.get("pmax", "")
        base_pmax = parse_float(case_pmax) if str(case_pmax).strip() else default_pmax
        if retry < args.fatal_retries:
            next_retry = retry + 1
            next_pmax = base_pmax / (10.0**next_retry)
            fresh_retry_by_slot[slot] = (case, next_retry)
            print(
                f"Case {case_id} will be retried by a fresh worker "
                f"({next_retry}/{args.fatal_retries}) with "
                f"pmax={next_pmax:g} bar: {reason}",
                flush=True,
            )
        else:
            mark_failed(
                task,
                f"{reason}; exhausted {args.fatal_retries} fatal retries",
            )

    def save_fatal_diagnostic(
        token: str,
        task: tuple[dict[str, str], int],
    ) -> None:
        case, retry = task
        case_id = str(case["case_id"])
        case_pmax = case.get("pmax", "")
        base_pmax = parse_float(case_pmax) if str(case_pmax).strip() else default_pmax
        pmax_used = base_pmax / (10.0**retry)
        fatal_path = work_root / token / "fatal.case"
        failure_dir = (
            output_root
            / "fatal_cases"
            / f"case_{case_id}_retry_{retry}_{token}"
        )
        failure_dir.mkdir(parents=True, exist_ok=True)
        if fatal_path.exists():
            shutil.copy2(fatal_path, failure_dir / "fatal.case")
        (failure_dir / "parameters.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in case.items())
            + f"\nfatal_retry={retry}\n"
            + f"base_pmax={base_pmax:.16g}\n"
            + f"pmax_used={pmax_used:.16g}\n"
        )

    def handle_message(message: tuple[Any, ...]) -> None:
        kind, token, *values = message
        info = workers.get(token)
        if info is None:
            return
        slot = int(info["slot"])

        if kind == "ready":
            ready.add(token)
            return

        if kind == "started":
            task = assigned.pop(token, None)
            if task is not None:
                inflight[token] = task
            return

        if kind == "done":
            case_id, retry, converged, destination = values
            inflight.pop(token, None)
            assigned.pop(token, None)
            completed.add(str(case_id))
            checkpoint(str(case_id), bool(converged))
            print(
                f"Finished case {case_id}; converged={converged}; "
                f"output={destination}",
                flush=True,
            )
            return

        if kind in {"fatal", "error"}:
            case_id, retry, details = values
            task = inflight.pop(token, None) or assigned.pop(token, None)
            ready.discard(token)
            print(
                f"{kind.upper()} in {token}, case {case_id}:\n{details}",
                file=sys.stderr,
                flush=True,
            )
            if task is not None:
                if kind == "fatal":
                    retry_or_fail(slot, task, kind)
                else:
                    mark_failed(task, kind)
            return

        if kind == "init_error":
            (details,) = values
            ready.discard(token)
            print(
                f"Initialization failed in {token}:\n{details}",
                file=sys.stderr,
                flush=True,
            )

    def drain_messages() -> None:
        """Drain every worker's private result pipe without blocking."""
        for token, info in list(workers.items()):
            connection = info["result"]
            while True:
                try:
                    if not connection.poll():
                        break
                    handle_message(connection.recv())
                except (EOFError, OSError):
                    break

    def dispatch() -> None:
        for token in list(ready):
            info = workers.get(token)
            if info is None or not info["process"].is_alive():
                ready.discard(token)
                continue
            slot = int(info["slot"])
            if slot in fresh_retry_by_slot:
                task = fresh_retry_by_slot.pop(slot)
            elif pending:
                task = pending.popleft()
            else:
                continue
            info["queue"].put(task)
            assigned[token] = task
            ready.discard(token)

    for slot in range(args.workers):
        start_worker(slot)

    total = len(selected)
    try:
        while len(completed) + len(permanently_failed) < total:
            drain_messages()

            # Python cannot inspect fatal.case while it is blocked inside the
            # Fortran pycomputemodel() call, so the supervisor watches every
            # active worker directory.  Polling occurs about every 0.2 s.
            for token, info in list(workers.items()):
                if token in forced_fatal:
                    continue
                task = inflight.get(token) or assigned.get(token)
                if task is None:
                    continue
                fatal_path = work_root / token / "fatal.case"
                if not fatal_path.exists():
                    continue

                case, retry = task
                case_id = str(case["case_id"])
                forced_fatal[token] = task
                inflight.pop(token, None)
                assigned.pop(token, None)
                ready.discard(token)
                print(
                    f"Terminating {token} immediately after fatal.case "
                    f"appeared for case {case_id}.",
                    flush=True,
                )
                info["process"].terminate()

            dead_tokens = [
                token
                for token, info in workers.items()
                if not info["process"].is_alive()
            ]
            for token in dead_tokens:
                workers[token]["process"].join()

            # Drain each dead worker's private result pipe once more after join
            # so a final fatal/error message is handled before recovery.
            drain_messages()

            for token in dead_tokens:
                info = workers.pop(token, None)
                if info is None:
                    continue
                slot = int(info["slot"])
                exitcode = info["process"].exitcode
                ready.discard(token)
                task = inflight.pop(token, None) or assigned.pop(token, None)
                info["queue"].close()
                info["result"].close()
                fatal_task = forced_fatal.pop(token, None)
                if fatal_task is not None:
                    save_fatal_diagnostic(token, fatal_task)
                    retry_or_fail(
                        slot,
                        fatal_task,
                        "fatal.case detected by supervisor",
                    )
                elif task is not None:
                    fatal_path = work_root / token / "fatal.case"
                    if fatal_path.exists():
                        save_fatal_diagnostic(token, task)
                        retry_or_fail(
                            slot,
                            task,
                            f"fatal.case found after worker exit {exitcode}",
                        )
                    else:
                        mark_failed(task, f"worker exited with code {exitcode}")

                if len(completed) + len(permanently_failed) < total:
                    start_worker(slot)

            dispatch()
            # Keep fatal.case detection responsive without busy-spinning.
            time.sleep(0.2)

        print(
            f"Grid finished: {len(completed)} completed, "
            f"{len(permanently_failed)} permanently failed.",
            flush=True,
        )
        return 1 if permanently_failed else 0

    finally:
        for token, info in list(workers.items()):
            if info["process"].is_alive():
                try:
                    info["queue"].put(None)
                except (BrokenPipeError, EOFError):
                    pass
        for info in workers.values():
            info["process"].join(timeout=10)
            if info["process"].is_alive():
                info["process"].terminate()
                info["process"].join()
            info["queue"].close()
            info["result"].close()


if __name__ == "__main__":
    raise SystemExit(main())
