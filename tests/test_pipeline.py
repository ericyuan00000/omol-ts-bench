"""Tests for run_pipeline using mocked stage functions (no UMA required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from omol_ts_bench.core.irc import IRCResult
from omol_ts_bench.core.neb import NEBResult
from omol_ts_bench.core.pipeline import PipelineResult, run_pipeline
from omol_ts_bench.core.sella import SellaResult
from omol_ts_bench.core.validate import ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


def _neb_result(converged=True):
    return NEBResult(
        converged=converged,
        ts_guess=_atoms() if converged else None,
        ts_energy=-1.0 if converged else None,
        barrier_forward=0.5 if converged else None,
        barrier_reverse=0.3 if converged else None,
        n_steps=10,
    )


def _sella_result(converged=True):
    return SellaResult(
        converged=converged,
        ts_optimized=_atoms() if converged else None,
        ts_energy=-1.1 if converged else None,
        imaginary_freq=-300.0 if converged else None,
        n_steps=8,
    )


def _irc_result(fwd=True, rev=True):
    return IRCResult(
        converged_forward=fwd,
        converged_reverse=rev,
        endpoint_forward=_atoms() if fwd else None,
        endpoint_reverse=_atoms() if rev else None,
        path_forward=None,
        path_reverse=None,
        n_steps_forward=20 if fwd else 0,
        n_steps_reverse=18 if rev else 0,
    )


def _validation_result(both=True):
    return ValidationResult(
        reactant_match=both,
        product_match=both,
        both_match=both,
    )


_PIPELINE_PATCHES = [
    "omol_ts_bench.core.pipeline.run_neb",
    "omol_ts_bench.core.pipeline.run_sella",
    "omol_ts_bench.core.pipeline.run_irc",
    "omol_ts_bench.core.pipeline.validate_irc_endpoints",
]


# ---------------------------------------------------------------------------
# Full success path
# ---------------------------------------------------------------------------

def test_pipeline_success():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()) as mock_neb,
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()) as mock_sella,
        patch(_PIPELINE_PATCHES[2], return_value=_irc_result()) as mock_irc,
        patch(_PIPELINE_PATCHES[3], return_value=_validation_result()) as mock_val,
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert isinstance(result, PipelineResult)
    assert result.error is None
    assert result.double_ended_result is not None
    assert result.sella_result is not None
    assert result.irc_result is not None
    assert result.validation_result is not None
    assert result.validation_result.both_match is True


def test_pipeline_returns_reaction_id():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], return_value=_irc_result()),
        patch(_PIPELINE_PATCHES[3], return_value=_validation_result()),
    ):
        result = run_pipeline("my_rxn", _atoms(), _atoms(), MagicMock())

    assert result.reaction_id == "my_rxn"


# ---------------------------------------------------------------------------
# NEB failure paths
# ---------------------------------------------------------------------------

def test_pipeline_neb_not_converged():
    with patch(_PIPELINE_PATCHES[0], return_value=_neb_result(converged=False)):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert "converge" in result.error.lower()
    assert result.sella_result is None
    assert result.irc_result is None


def test_pipeline_neb_exception():
    with patch(_PIPELINE_PATCHES[0], side_effect=RuntimeError("neb blew up")):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert result.double_ended_result is None


# ---------------------------------------------------------------------------
# Sella failure paths
# ---------------------------------------------------------------------------

def test_pipeline_sella_not_converged():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result(converged=False)),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert result.double_ended_result is not None
    assert result.irc_result is None


def test_pipeline_sella_exception():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], side_effect=RuntimeError("sella blew up")),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert "Sella" in result.error


# ---------------------------------------------------------------------------
# IRC failure paths
# ---------------------------------------------------------------------------

def test_pipeline_irc_missing_endpoint():
    bad_irc = _irc_result(fwd=True, rev=False)
    bad_irc.endpoint_reverse = None

    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], return_value=bad_irc),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert result.validation_result is None


def test_pipeline_irc_exception():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], side_effect=RuntimeError("irc failed")),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert "IRC" in result.error


# ---------------------------------------------------------------------------
# Validation failure paths
# ---------------------------------------------------------------------------

def test_pipeline_validation_mismatch():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], return_value=_irc_result()),
        patch(_PIPELINE_PATCHES[3], return_value=_validation_result(both=False)),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    # No error — mismatch is a result, not a failure
    assert result.error is None
    assert result.validation_result.both_match is False


def test_pipeline_validation_exception():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], return_value=_irc_result()),
        patch(_PIPELINE_PATCHES[3], side_effect=RuntimeError("openbabel crash")),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock())

    assert result.error is not None
    assert "Validation" in result.error


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

def test_pipeline_unknown_method():
    result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock(),
                          double_ended_method="unknown")
    assert result.error is not None
    assert "Unknown" in result.error


def test_pipeline_method_recorded():
    with (
        patch(_PIPELINE_PATCHES[0], return_value=_neb_result()),
        patch(_PIPELINE_PATCHES[1], return_value=_sella_result()),
        patch(_PIPELINE_PATCHES[2], return_value=_irc_result()),
        patch(_PIPELINE_PATCHES[3], return_value=_validation_result()),
    ):
        result = run_pipeline("rxn_001", _atoms(), _atoms(), MagicMock(),
                              double_ended_method="neb")

    assert result.double_ended_method == "neb"
