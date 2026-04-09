"""Calculator setup for UMA with OMol head."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


def get_uma_calculator(
    model_name: str = "uma-s-1p2",
    task_name: str = "omol",
    **kwargs,
) -> Calculator:
    """Get UMA small calculator with OMol head.

    Parameters
    ----------
    model_name
        Pretrained model name. Available UMA models:
        "uma-s-1p2" (small, v1.2), "uma-s-1p1" (small, v1.1),
        "uma-m-1p1" (medium, v1.1).
    task_name
        Task/head name, e.g. "omol" for organometallic reactions.
    **kwargs
        Additional keyword arguments passed to ``get_predict_unit``.

    Returns
    -------
    Calculator
        ASE-compatible calculator instance.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predict_unit = pretrained_mlip.get_predict_unit(model_name, **kwargs)
    calc = FAIRChemCalculator(predict_unit, task_name=task_name)
    return calc
