"""End-to-end pipeline: endpoints -> NEB/GSM -> Sella -> IRC -> validation.

This module composes the individual steps into a single function that
runs the full benchmark pipeline for one reaction. Workflow adapters
call this function (or its components) to parallelize across reactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omol_ts_bench.core.irc import IRCResult, run_irc
from omol_ts_bench.core.neb import NEBResult, run_neb
from omol_ts_bench.core.sella import SellaResult, run_sella
from omol_ts_bench.core.validate import ValidationResult, validate_irc_endpoints

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class PipelineResult:
    """Full result of the benchmark pipeline for one reaction."""

    reaction_id: str
    double_ended_method: str
    double_ended_result: NEBResult | None
    sella_result: SellaResult | None
    irc_result: IRCResult | None
    validation_result: ValidationResult | None
    error: str | None = None


def run_pipeline(
    reaction_id: str,
    reactant: Atoms,
    product: Atoms,
    calc: Calculator,
    double_ended_method: str = "neb",
    neb_kwargs: dict | None = None,
    sella_kwargs: dict | None = None,
    irc_kwargs: dict | None = None,
) -> PipelineResult:
    """Run the full TS benchmark pipeline for a single reaction.

    Steps:
        1. Double-ended search (NEB or GSM) to find a TS guess
        2. Sella saddle-point optimization to refine the TS
        3. IRC to verify the TS connects the correct endpoints
        4. OpenBabel validation to check IRC endpoints match inputs

    Parameters
    ----------
    reaction_id
        Unique identifier for this reaction.
    reactant
        Reactant endpoint geometry.
    product
        Product endpoint geometry.
    calc
        ASE calculator instance.
    double_ended_method
        "neb" or "gsm".
    neb_kwargs
        Keyword arguments for run_neb.
    sella_kwargs
        Keyword arguments for run_sella.
    irc_kwargs
        Keyword arguments for run_irc.

    Returns
    -------
    PipelineResult
        Complete results from all pipeline stages.
    """
    neb_kwargs = neb_kwargs or {}
    sella_kwargs = sella_kwargs or {}
    irc_kwargs = irc_kwargs or {}

    # Step 1: Double-ended TS search
    de_result = None
    try:
        if double_ended_method == "neb":
            de_result = run_neb(reactant, product, calc, **neb_kwargs)
        elif double_ended_method == "gsm":
            from omol_ts_bench.core.gsm import run_de_gsm
            de_result = run_de_gsm(reactant, product, calc)
        else:
            return PipelineResult(
                reaction_id=reaction_id,
                double_ended_method=double_ended_method,
                double_ended_result=None, sella_result=None,
                irc_result=None, validation_result=None,
                error=f"Unknown double-ended method: {double_ended_method}",
            )
    except Exception as e:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=None, sella_result=None,
            irc_result=None, validation_result=None,
            error=f"Double-ended search failed: {e}",
        )

    if not de_result.converged or de_result.ts_guess is None:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=None,
            irc_result=None, validation_result=None,
            error="Double-ended search did not converge",
        )

    # Step 2: Sella TS refinement
    sella_result = None
    try:
        sella_result = run_sella(de_result.ts_guess, calc, **sella_kwargs)
    except Exception as e:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=None,
            irc_result=None, validation_result=None,
            error=f"Sella optimization failed: {e}",
        )

    if not sella_result.converged or sella_result.ts_optimized is None:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=sella_result,
            irc_result=None, validation_result=None,
            error="Sella optimization did not converge",
        )

    # Step 3: IRC
    irc_result = None
    try:
        irc_result = run_irc(sella_result.ts_optimized, calc, **irc_kwargs)
    except Exception as e:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=sella_result,
            irc_result=None, validation_result=None,
            error=f"IRC failed: {e}",
        )

    if irc_result.endpoint_forward is None or irc_result.endpoint_reverse is None:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=sella_result,
            irc_result=irc_result, validation_result=None,
            error="IRC did not produce both endpoints",
        )

    # Step 4: Validation
    validation_result = None
    try:
        validation_result = validate_irc_endpoints(
            reactant, product,
            irc_result.endpoint_forward, irc_result.endpoint_reverse,
        )
    except Exception as e:
        return PipelineResult(
            reaction_id=reaction_id,
            double_ended_method=double_ended_method,
            double_ended_result=de_result, sella_result=sella_result,
            irc_result=irc_result, validation_result=None,
            error=f"Validation failed: {e}",
        )

    return PipelineResult(
        reaction_id=reaction_id,
        double_ended_method=double_ended_method,
        double_ended_result=de_result,
        sella_result=sella_result,
        irc_result=irc_result,
        validation_result=validation_result,
    )
