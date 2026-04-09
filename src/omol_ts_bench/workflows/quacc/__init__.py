"""QuAcc workflow adapter for HPC-scale benchmarking.

Uses QuAcc decorators to distribute reactions across compute nodes.
Designed for NERSC Perlmutter but works with any QuAcc-supported
workflow engine (Parsl, Dask, Prefect, Covalent, etc.).

Setup:
    1. Configure QuAcc workflow engine:
       $ quacc set WORKFLOW_ENGINE parsl  # or dask, prefect, etc.

    2. Configure compute resources in your QuAcc settings or
       provide a Parsl/Dask config.

See https://quantum-accelerators.github.io/quacc/ for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from omol_ts_bench.core.irc import IRCResult
    from omol_ts_bench.core.neb import NEBResult
    from omol_ts_bench.core.pipeline import PipelineResult
    from omol_ts_bench.core.sella import SellaResult


def make_quacc_pipeline(
    calc_factory: callable,
    double_ended_method: str = "neb",
    neb_kwargs: dict | None = None,
    sella_kwargs: dict | None = None,
    irc_kwargs: dict | None = None,
):
    """Create QuAcc-decorated pipeline functions.

    Returns decorated versions of the pipeline steps that can be
    dispatched by QuAcc's workflow engine.

    Parameters
    ----------
    calc_factory
        Callable that returns an ASE Calculator.
    double_ended_method
        "neb" or "gsm".
    neb_kwargs
        Keyword arguments for NEB.
    sella_kwargs
        Keyword arguments for Sella.
    irc_kwargs
        Keyword arguments for IRC.

    Returns
    -------
    callable
        A QuAcc-compatible job function that runs the full pipeline.
    """
    from quacc import job

    neb_kwargs = neb_kwargs or {}
    sella_kwargs = sella_kwargs or {}
    irc_kwargs = irc_kwargs or {}

    @job
    def pipeline_job(
        reaction_id: str,
        reactant: Atoms,
        product: Atoms,
    ) -> dict:
        """Run full pipeline as a single QuAcc job."""
        from omol_ts_bench.core.pipeline import run_pipeline

        calc = calc_factory()
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

        # Return serializable summary
        return {
            "reaction_id": result.reaction_id,
            "error": result.error,
            "neb_converged": result.double_ended_result.converged if result.double_ended_result else None,
            "sella_converged": result.sella_result.converged if result.sella_result else None,
            "irc_converged_forward": result.irc_result.converged_forward if result.irc_result else None,
            "irc_converged_reverse": result.irc_result.converged_reverse if result.irc_result else None,
            "validation_both_match": result.validation_result.both_match if result.validation_result else None,
            "ts_energy": result.sella_result.ts_energy if result.sella_result else None,
            "barrier_forward": result.double_ended_result.barrier_forward if result.double_ended_result else None,
        }

    return pipeline_job


def run_benchmark_quacc(
    reactions: list,
    calc_factory: callable,
    double_ended_method: str = "neb",
    neb_kwargs: dict | None = None,
    sella_kwargs: dict | None = None,
    irc_kwargs: dict | None = None,
) -> list:
    """Run the benchmark using QuAcc's workflow engine.

    Parameters
    ----------
    reactions
        List of Reaction objects.
    calc_factory
        Callable that returns an ASE Calculator.
    double_ended_method
        "neb" or "gsm".
    neb_kwargs
        Keyword arguments for NEB.
    sella_kwargs
        Keyword arguments for Sella.
    irc_kwargs
        Keyword arguments for IRC.

    Returns
    -------
    list
        List of futures/results depending on the workflow engine.
    """
    pipeline_job = make_quacc_pipeline(
        calc_factory=calc_factory,
        double_ended_method=double_ended_method,
        neb_kwargs=neb_kwargs,
        sella_kwargs=sella_kwargs,
        irc_kwargs=irc_kwargs,
    )

    futures = []
    for rxn in reactions:
        future = pipeline_job(
            reaction_id=rxn.reaction_id,
            reactant=rxn.reactant,
            product=rxn.product,
        )
        futures.append(future)

    return futures
