"""Run AMOSA with Numba operations and annual vehicle/fuel-cycle emissions."""

from __future__ import annotations

import argparse
from itertools import islice, product
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
import time

import numpy as np
import pandas as pd

from problem import build_city_problem
from vehicle_type import EMISSION_METHOD, GRID_G_CO2E_PER_KWH, CHARGING_EFFICIENCY
from evaluator import EvaluationSettings, FastAMOSAEvaluator
from mosa import MOSA
from paths import SHARED_DATA_ROOT
import sim_utils
from vehicle_type import vehi_type_num


DEFAULT_SEED = 42


def additional_references(cs_num: int, seed: int, limit: int):
    rng = np.random.default_rng(seed)
    model_counts = [vehi_type_num(kind) for kind in ("large", "medium", "small")]
    tried = 0
    for models in product(*(range(count) for count in model_counts)):
        if tried >= limit:
            return
        if models == (0, 0, 0):
            continue
        yield np.r_[np.ones(cs_num, dtype=np.int32), models]
        tried += 1
    while tried < limit:
        density = rng.choice((0.25, 0.5, 0.75, 0.9, 1.0))
        stations = (rng.random(cs_num) < density).astype(np.int32)
        if not stations.any():
            stations[rng.integers(cs_num)] = 1
        models = [rng.integers(count) for count in model_counts]
        yield np.r_[stations, models]
        tried += 1


def load_feasible_references(algorithm, evaluator, initial, cs_num, seed, limit):
    try:
        algorithm.load_refer_solutions(initial)
        return len(initial)
    except ValueError as exc:
        if str(exc) != "No feasible reference solution was found":
            raise
    attempted = len(initial)
    candidates = additional_references(cs_num, seed, limit)
    while batch := list(islice(candidates, 16)):
        population = np.asarray(batch, dtype=np.int32)
        objectives, diagnostics = evaluator.evaluate_population(population)
        attempted += len(population)
        if any(algorithm.is_feasible(cv) for cv in diagnostics):
            algorithm.load_refer_solutions(population, objectives, diagnostics)
            return attempted
    raise ValueError(f"No feasible reference solution found in {attempted} attempts")


def run_city(city: str, args) -> str:
    try:
        output = args.output_root / city
        required = ("archive_vars.csv", "archive_objs.csv", "archive_cvs.csv", "hv_history.csv")
        if (not args.overwrite and (output / "run_config.json").is_file()
                and all((output / name).is_file() and (output / name).stat().st_size > 0
                        for name in required)):
            saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            if saved.get("hvac_power") != args.hvac_power:
                raise ValueError("Existing output uses a different HVAC power; use a new output root")
            if saved.get("emission_adjustment") != EMISSION_METHOD:
                raise ValueError("Existing output uses an older emission calculation; use --overwrite")
            return f"SKIP {city}: complete output already exists at {output}"

        city_seed = sim_utils.derive_seed(args.seed, city)
        sim_utils.set_random_seed(city_seed)
        problem = build_city_problem(city, reserve_energy=args.reserve_energy, vs_root=args.vs_root)
        settings = EvaluationSettings(
            reserve_energy=args.reserve_energy,
            min_interval=args.min_interval,
            degradation_factor=args.degradation_factor,
            degradation_share=args.degradation_share,
            degraded_capacity_factor=args.degraded_capacity_factor,
            degradation_seed=city_seed,
            battery_purchase_factor=args.battery_purchase_factor,
            vehicle_price_factor=args.vehicle_price_factor,
            power_factor=args.power_factor,
            slow_power_factor=args.slow_power_factor,
            fast_charger_cost_factor=args.fast_charger_cost_factor,
            e_price_factor=args.e_price_factor,
            fast_chargers=args.fast_chargers,
            slow_chargers=args.slow_chargers,
            energy_factor=args.energy_factor,
            hvac_power=args.hvac_power,
            adjust_energy_emissions=args.adjust_energy_emissions,
            charging_profile=args.charging_profile,
        )
        evaluator = FastAMOSAEvaluator(problem, settings)
        algorithm = MOSA(
            problem,
            annealing_iters=args.annealing_iters,
            annealing_strength=args.annealing_strength,
            cooling_alpha=args.cooling_alpha,
            evaluator=evaluator,
            vehicle_perturb_scale=args.vehicle_perturb_scale,
        )
        algorithm.min_delta_hv = args.min_delta_hv
        algorithm.what_if = False
        algorithm.multiprocess = False
        algorithm.early_termination = {
            "max_iters": args.max_iters,
            "max_duration": args.max_duration,
            "max_no_eliminated": args.max_no_eliminated,
        }

        refer_vars = sim_utils.set_refer_vars(problem.cs_num, refer_num=args.references)
        started = time.perf_counter()
        attempted_references = load_feasible_references(
            algorithm, evaluator, refer_vars, problem.cs_num, city_seed,
            args.additional_references,
        )

        output.mkdir(parents=True, exist_ok=True)
        marker = output / "run_config.json"
        if marker.exists():
            marker.unlink()
        algorithm.run(is_cv=True, path=str(output), store="store")
        elapsed = time.perf_counter() - started
        hv_values = algorithm.output_fitness()
        pd.DataFrame(
            {
                "cooling_step": range(1, len(hv_values) + 1),
                "evaluations": [
                    step * algorithm.annealing_iters
                    for step in range(1, len(hv_values) + 1)
                ],
                "normalized_hv": hv_values,
            }
        ).to_csv(output / "hv_history.csv", index=False)

        config = {
            "model": "China eBus",
            "optimizer": "AMOSA-ANS",
            "evaluator": "Numba operations with vehicle/fuel-cycle emissions",
            "city": city,
            "base_seed": args.seed,
            "city_seed": city_seed,
            "vs_root": str(args.vs_root) if args.vs_root is not None else None,
            "sa_parameter": getattr(args, "sa_parameter", None),
            "sa_value": getattr(args, "sa_value", None),
            "what_if_parameter": getattr(args, "what_if_parameter", None),
            "what_if_value": getattr(args, "what_if_value", None),
            "references": args.references,
            "attempted_references": attempted_references,
            "initial_temperature": algorithm.initial_temperature,
            "end_temperature": algorithm.end_temperature,
            "cooling_alpha": algorithm.cooling_alpha,
            "annealing_iters": algorithm.annealing_iters,
            "annealing_strength": algorithm.annealing_strength,
            "vehicle_perturb_scale": algorithm.vehicle_perturb_scale,
            "min_delta_hv": algorithm.min_delta_hv,
            "max_iters": args.max_iters,
            "max_duration_hours": args.max_duration,
            "max_no_eliminated": args.max_no_eliminated,
            "reserve_energy": args.reserve_energy,
            "min_interval": args.min_interval,
            "degradation_factor": args.degradation_factor,
            "degradation_share": args.degradation_share,
            "degraded_capacity_factor": args.degraded_capacity_factor,
            "battery_purchase_factor": args.battery_purchase_factor,
            "vehicle_price_factor": args.vehicle_price_factor,
            "power_factor": args.power_factor,
            "slow_power_factor": args.slow_power_factor,
            "fast_charger_cost_factor": args.fast_charger_cost_factor,
            "e_price_factor": args.e_price_factor,
            "fast_chargers": args.fast_chargers,
            "slow_chargers": args.slow_chargers,
            "energy_factor": args.energy_factor,
            "hvac_power": args.hvac_power,
            "emission_adjustment": EMISSION_METHOD,
            "grid_g_co2e_per_kwh": GRID_G_CO2E_PER_KWH,
            "charging_efficiency": CHARGING_EFFICIENCY,
            "vehicle_cycle_lifespan_years": 10,
            "battery_growth_vehicle_cycle_fixed": True,
            "adjust_energy_emissions": args.adjust_energy_emissions,
            "charging_profile": args.charging_profile,
            "elapsed_seconds": elapsed,
            "archive_size": len(algorithm.archive),
            "cooling_steps": len(hv_values),
        }
        temporary_marker = output / "run_config.json.tmp"
        temporary_marker.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_marker.replace(marker)
        return f"OK {city}: {elapsed:.1f}s, archive={len(algorithm.archive)}"
    except Exception as exc:
        return f"FAILED {city}: {exc}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--cities-file", type=Path, default=SHARED_DATA_ROOT / "224cities.csv")
    parser.add_argument("--vs-root", type=Path)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--references", type=int, default=20)
    parser.add_argument("--additional-references", type=int, default=512)
    parser.add_argument("--annealing-iters", type=int, default=100)
    parser.add_argument("--annealing-strength", type=float, default=0.6)
    parser.add_argument("--vehicle-perturb-scale", type=float, default=0.5)
    parser.add_argument("--cooling-alpha", type=float, default=0.9953)
    parser.add_argument("--min-delta-hv", type=float, default=0.005)
    parser.add_argument("--max-iters", type=int, default=500000)
    parser.add_argument("--max-duration", type=int, default=240, help="Hours per city")
    parser.add_argument("--max-no-eliminated", type=int, default=30000)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--min-interval", type=float, default=60.0)
    parser.add_argument("--degradation-factor", type=float, default=1.0)
    parser.add_argument("--degradation-share", type=float, default=0.0)
    parser.add_argument("--degraded-capacity-factor", type=float, default=0.9)
    parser.add_argument("--battery-purchase-factor", type=float, default=1.0)
    parser.add_argument("--vehicle-price-factor", type=float, default=1.0)
    parser.add_argument("--power-factor", type=float, default=1.0)
    parser.add_argument("--slow-power-factor", type=float)
    parser.add_argument("--fast-charger-cost-factor", type=float, default=1.0)
    parser.add_argument("--e-price-factor", type=float, default=1.0)
    parser.add_argument("--fast-chargers", type=int, default=20)
    parser.add_argument("--slow-chargers", type=int, default=20)
    parser.add_argument("--energy-factor", type=float, default=1.0)
    parser.add_argument("--adjust-energy-emissions", action="store_true",
                        help="Compatibility flag; full energy emissions are always included")
    parser.add_argument("--hvac-power", type=float, default=EvaluationSettings().hvac_power,
                        help="HVAC power in kW; -1 uses the stored 20 kW reference")
    parser.add_argument("--charging-profile", choices=("linear", "part_e_both"), default="linear")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SHARED_DATA_ROOT / "output" / "results" / "baseline",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.cities:
        cities = args.cities
    else:
        cities = pd.read_csv(args.cities_file)["city"].tolist()
    if len(cities) != len(set(cities)):
        raise ValueError("Duplicate city names are not allowed")
    worker = partial(run_city, args=args)
    results = []
    if args.processes == 1:
        for index, city in enumerate(cities, start=1):
            result = worker(city)
            results.append(result)
            print(f"[{index}/{len(cities)}] {result}", flush=True)
    else:
        with mp.get_context("spawn").Pool(min(args.processes, len(cities))) as pool:
            for index, result in enumerate(pool.imap_unordered(worker, cities, chunksize=1), start=1):
                results.append(result)
                print(f"[{index}/{len(cities)}] {result}", flush=True)
    if any(result.startswith("FAILED") for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
