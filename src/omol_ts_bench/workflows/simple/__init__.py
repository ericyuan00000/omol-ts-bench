"""Simple local workflow adapter using multiprocessing.

This adapter requires no external workflow engine. It runs the benchmark
pipeline either serially or in parallel using Python's multiprocessing.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from omol_ts_bench.data.loader import Reaction

logger = logging.getLogger(__name__)


def _run_single_reaction(
    reaction_id: str,
    reactant_path: str,
    product_path: str,
    calc_factory: str,
    double_ended_method: str,
    neb_kwargs: dict,
    sella_kwargs: dict,
    irc_kwargs: dict,
) -> dict:
    """Worker function for multiprocessing (must be picklable)."""
    from ase.io import read

    from omol_ts_bench.core.pipeline import run_pipeline

    # Reconstruct calculator from factory function path
    module_path, func_name = calc_factory.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    calc = getattr(mod, func_name)()

    reactant = read(reactant_path)
    product = read(product_path)

    result = run_pipeline(
        reaction_id=reaction_id,
        reactant=reactant,
        product=product,
        calc=calc,
        double_ended_method=double_ended_method,
        neb_kwargs=neb_kwargs,
        sella_kwargs=sella_kwargs,
        irc_kwargs=irc_kwargs,
    )

    # Convert to serializable dict (skip Atoms objects)
    summary = {
        "reaction_id": result.reaction_id,
        "double_ended_method": result.double_ended_method,
        "error": result.error,
        "neb_converged": result.double_ended_result.converged if result.double_ended_result else None,
        "sella_converged": result.sella_result.converged if result.sella_result else None,
        "irc_converged_forward": result.irc_result.converged_forward if result.irc_result else None,
        "irc_converged_reverse": result.irc_result.converged_reverse if result.irc_result else None,
        "validation_both_match": result.validation_result.both_match if result.validation_result else None,
        "ts_energy": result.sella_result.ts_energy if result.sella_result else None,
        "barrier_forward": result.double_ended_result.barrier_forward if result.double_ended_result else None,
    }
    return summary


def run_benchmark(
    reactions: list[Reaction],
    calc_factory: str = "omol_ts_bench.core.calculator.get_uma_calculator",
    double_ended_method: str = "neb",
    n_workers: int = 1,
    output_dir: str | Path = "results",
    neb_kwargs: dict | None = None,
    sella_kwargs: dict | None = None,
    irc_kwargs: dict | None = None,
) -> list[dict]:
    """Run the benchmark on a list of reactions.

    Parameters
    ----------
    reactions
        List of Reaction objects to process.
    calc_factory
        Dotted path to a callable that returns an ASE Calculator.
        Must be importable in worker processes.
    double_ended_method
        "neb" or "gsm".
    n_workers
        Number of parallel workers. 1 = serial execution.
    output_dir
        Directory to write results JSON.
    neb_kwargs
        Keyword arguments for NEB.
    sella_kwargs
        Keyword arguments for Sella.
    irc_kwargs
        Keyword arguments for IRC.

    Returns
    -------
    list[dict]
        Summary results for each reaction.
    """
    from ase.io import write

    neb_kwargs = neb_kwargs or {}
    sella_kwargs = sella_kwargs or {}
    irc_kwargs = irc_kwargs or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write temp XYZ files for multiprocessing (workers need file paths)
    tmp_dir = output_dir / ".tmp_xyz"
    tmp_dir.mkdir(exist_ok=True)

    tasks = []
    for rxn in reactions:
        r_path = tmp_dir / f"{rxn.reaction_id}_reactant.xyz"
        p_path = tmp_dir / f"{rxn.reaction_id}_product.xyz"
        write(str(r_path), rxn.reactant)
        write(str(p_path), rxn.product)
        tasks.append((rxn.reaction_id, str(r_path), str(p_path)))

    results = []

    if n_workers <= 1:
        # Serial execution
        for rxn_id, r_path, p_path in tasks:
            logger.info("Running reaction %s", rxn_id)
            result = _run_single_reaction(
                rxn_id, r_path, p_path, calc_factory,
                double_ended_method, neb_kwargs, sella_kwargs, irc_kwargs,
            )
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for rxn_id, r_path, p_path in tasks:
                future = executor.submit(
                    _run_single_reaction,
                    rxn_id, r_path, p_path, calc_factory,
                    double_ended_method, neb_kwargs, sella_kwargs, irc_kwargs,
                )
                futures[future] = rxn_id

            for future in as_completed(futures):
                rxn_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info("Completed %s: match=%s", rxn_id, result.get("validation_both_match"))
                except Exception as e:
                    logger.error("Failed %s: %s", rxn_id, e)
                    results.append({"reaction_id": rxn_id, "error": str(e)})

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_path)

    return results
