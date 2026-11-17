"""Run sensitivity and What-if scenarios using prepared city inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd

from paths import SHARED_DATA_ROOT
from run_amosa import parse_args as parse_model_args, run_city, validate_run_config


@dataclass(frozen=True)
class Case:
    group: str
    name: str
    parameter: str
    value: float | int | str

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"


CASES = (
    *(Case("operating_time", name, "operating_factor", value) for name, value in
      (("minus20", 0.8), ("minus10", 0.9), ("plus10", 1.1), ("plus20", 1.2))),
    *(Case("missing_timetable", name, "headway_factor", value) for name, value in
      (("minus20", 0.8), ("minus10", 0.9), ("plus10", 1.1), ("plus20", 1.2))),
    *(Case("battery_degradation", f"{value}pct", "degradation_share", value / 100)
      for value in (10, 20, 30, 40)),
    *(Case("charging_window", f"{value}min", "min_interval", value)
      for value in (48, 54, 66, 72)),
    *(Case("neighbourhood_size", str(value), "annealing_strength", value)
      for value in (0.2, 0.4, 0.8, 1.0)),
    *(Case("seasonal_hvac", f"{value}kw", "hvac_power", value)
      for value in (2, 3, 5, 6)),
    Case("nonlinear_charging", "part_e_both", "charging_profile", "part_e_both"),
)

WHAT_IF_CASES = (
    *(Case("fast_charging_speed", f"FCS{value}", "power_factor", value)
      for value in (2, 3, 4, 5)),
    *(Case("battery_capacity_growth", f"BC{value:g}", "degradation_factor", value)
      for value in (1.25, 1.5, 1.75, 2.0)),
    *(Case("vehicle_purchase_cost", f"VC{value}", "vehicle_price_factor", value)
      for value in (0.6, 0.7, 0.8, 0.9)),
    *(Case("fast_only_speed", f"FCS{value}", "power_factor", value)
      for value in (2, 3, 4, 5)),
    *(Case("fast_speed_with_cost", f"FCS{value}", "power_factor", value)
      for value in (2, 3, 4, 5)),
)

FAST_CHARGER_COST_FACTORS = {2: 93 / 48, 3: 173 / 48, 4: 279 / 48, 5: 458 / 48}


def _complete(output: Path) -> bool:
    files = ("run_config.json", "archive_vars.csv", "archive_objs.csv",
             "archive_cvs.csv", "hv_history.csv")
    return all((output / name).is_file() and (output / name).stat().st_size > 0
               for name in files)


def _matching_output(output: Path, case: Case, seed: int,
                     max_iters: int, max_duration: int, input_root: Path | None = None) -> bool:
    if not _complete(output):
        return False
    try:
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    args = _case_args(output.name, case, output.parent,
                      input_root or SHARED_DATA_ROOT / "input" / "SA",
                      seed, max_iters, max_duration)
    try:
        validate_run_config(config, output.name, args)
    except ValueError:
        return False
    return True


def _case_args(city: str, case: Case, output_root: Path, input_root: Path,
               seed: int, max_iters: int, max_duration: int):
    args = parse_model_args([])
    args.seed = seed
    args.max_iters = max_iters
    args.max_duration = max_duration
    args.output_root = output_root
    if case in WHAT_IF_CASES:
        args.what_if_parameter = case.parameter
        args.what_if_value = case.value
        if case.group == "battery_capacity_growth":
            args.battery_purchase_factor = case.value
        if case.group in ("fast_only_speed", "fast_speed_with_cost"):
            args.slow_power_factor = 1.0
            if case.group == "fast_speed_with_cost":
                args.fast_charger_cost_factor = FAST_CHARGER_COST_FACTORS[case.value]
    else:
        args.sa_parameter = case.parameter
        args.sa_value = case.value
    setattr(args, case.parameter, case.value)
    args.adjust_energy_emissions = False
    if case.group in ("operating_time", "missing_timetable"):
        args.vs_root = input_root / case.group / case.name / "vs_parking_nodeid"
    return args


def _worker(city: str, province: str, case: Case, output_root: Path,
            input_root: Path, seed: int, max_iters: int, max_duration: int) -> str:
    output = output_root / case.group / case.name / city
    if _matching_output(output, case, seed, max_iters, max_duration, input_root):
        return f"SKIP {city}: complete"
    if _complete(output):
        return f"FAILED {city}: existing output has different settings"
    args = _case_args(city, case, output.parent, input_root, seed, max_iters, max_duration)
    if args.vs_root is not None and not (args.vs_root / f"{city}.csv").is_file():
        return f"FAILED {city}: missing prepared SA input at {args.vs_root}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            return run_city(city, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=500000)
    parser.add_argument("--max-duration", type=int, default=240)
    parser.add_argument("--what-if", action="store_true")
    parser.add_argument("--include-what-if", action="store_true")
    parser.add_argument("--what-if-output-root", type=Path,
                        default=SHARED_DATA_ROOT / "output" / "results" / "what_if")
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--groups", nargs="+", choices=sorted({c.group for c in CASES + WHAT_IF_CASES}))
    parser.add_argument("--output-root", type=Path,
                        default=None)
    parser.add_argument("--input-root", type=Path,
                        default=SHARED_DATA_ROOT / "input" / "SA")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.what_if and args.include_what_if:
        parser.error("--what-if and --include-what-if cannot be combined")
    if args.output_root is None:
        args.output_root = SHARED_DATA_ROOT / "output" / "results" / ("what_if" if args.what_if else "sa")
    if args.processes < 1:
        parser.error("--processes must be positive")
    queue_lock = None
    if args.include_what_if and not args.dry_run and os.name == "posix":
        import fcntl
        queue_lock = (SHARED_DATA_ROOT.parent / ".combined_scenarios_queue.lock").open("a+")
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A combined scenario queue is already running in this project")
    cities = pd.read_csv(SHARED_DATA_ROOT / "224cities.csv").set_index("city")
    selected = args.cities or cities.index.tolist()
    if len(selected) != len(set(selected)) or any(city not in cities.index for city in selected):
        parser.error("Cities must be unique and present in 224cities.csv")
    selected_cases = WHAT_IF_CASES if args.what_if else CASES
    if args.include_what_if:
        selected_cases = tuple(c for c in WHAT_IF_CASES if c.group in (
            "fast_charging_speed", "battery_capacity_growth", "vehicle_purchase_cost")) + tuple(
                c for c in CASES if c.group == "seasonal_hvac") + tuple(
                c for c in CASES if c.group != "seasonal_hvac")
    cases = [case for case in selected_cases if args.groups is None or case.group in args.groups]
    if not cases:
        parser.error("No scenarios selected for this run mode")
    def root_for(case):
        return args.what_if_output_root if args.include_what_if and case in WHAT_IF_CASES else args.output_root

    tasks = [(city, cities.loc[city, "province"], case) for case in cases for city in selected
             if not _matching_output(root_for(case) / case.group / case.name / city,
                                     case, args.seed, args.max_iters, args.max_duration, args.input_root)]
    print(f"{len(cases)} cases x {len(selected)} cities; at most {args.processes} processes", flush=True)
    counts = {}
    for case in cases:
        done = sum(_matching_output(root_for(case) / case.group / case.name / city,
                                    case, args.seed, args.max_iters, args.max_duration, args.input_root)
                   for city in selected)
        counts[case.key] = done
        print(f"  {case.key}: {done}/{len(selected)} complete", flush=True)
    if args.dry_run:
        return

    for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    failures = []
    outstanding = {"what_if": 0, "sa": 0}
    for _, _, case in tasks:
        outstanding["what_if" if case in WHAT_IF_CASES else "sa"] += 1
    stage_failures = {"what_if": 0, "sa": 0}
    if args.include_what_if and not outstanding["what_if"]:
        print("WHAT_IF COMPLETE: results ready for download", flush=True)
    pending = {}
    task_iter = iter(tasks)
    if not tasks:
        print("All cases complete", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.processes,
                             mp_context=mp.get_context("spawn")) as pool:
        def submit_next() -> bool:
            try:
                city, province, case = next(task_iter)
            except StopIteration:
                return False
            future = pool.submit(_worker, city, province, case, root_for(case),
                                 args.input_root, args.seed, args.max_iters, args.max_duration)
            pending[future] = (city, case)
            return True

        for _ in range(min(args.processes, len(tasks))):
            submit_next()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                city, case = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = f"FAILED {city}: worker exception: {exc}"
                counts[case.key] += 1
                print(f"[{case.key} {counts[case.key]}/{len(selected)}] {result}", flush=True)
                if result.startswith("FAILED"):
                    failures.append(f"{case.key}: {result}")
                    stage_failures["what_if" if case in WHAT_IF_CASES else "sa"] += 1
                stage = "what_if" if case in WHAT_IF_CASES else "sa"
                outstanding[stage] -= 1
                if args.include_what_if and outstanding[stage] == 0:
                    status = "FAILED" if stage_failures[stage] else "COMPLETE"
                    print(f"{stage.upper()} {status}: {stage_failures[stage]} failed city-runs; results at "
                          f"{args.what_if_output_root if stage == 'what_if' else args.output_root}", flush=True)
                submit_next()
    if failures:
        raise SystemExit(f"{len(failures)} city-runs failed; inspect launcher log")


if __name__ == "__main__":
    main()
