"""Tests for run_irc using a mocked Sella IRC (no UMA required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from omol_ts_bench.core.irc import IRCResult, run_irc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


def _make_calc():
    calc = MagicMock()
    calc.get_potential_energy.return_value = -1.0
    calc.get_forces.return_value = np.zeros((2, 3))
    calc.results = {"energy": -1.0, "forces": np.zeros((2, 3))}
    return calc


def _mock_irc_instance(converged=True, nsteps=10):
    mock = MagicMock()
    mock.run.return_value = converged
    mock.nsteps = nsteps
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("sella.IRC")
def test_run_irc_returns_irc_result(MockIRC):
    MockIRC.return_value = _mock_irc_instance(converged=True)

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc)
    assert isinstance(result, IRCResult)


@patch("sella.IRC")
def test_run_irc_both_directions_by_default(MockIRC):
    MockIRC.return_value = _mock_irc_instance(converged=True)

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="both")
    # IRC should have been instantiated twice (forward + reverse)
    assert MockIRC.call_count == 2


@patch("sella.IRC")
def test_run_irc_forward_only(MockIRC):
    MockIRC.return_value = _mock_irc_instance(converged=True)

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="forward")
    assert MockIRC.call_count == 1
    assert result.converged_forward is True
    # reverse was not run — converged_reverse stays False
    assert result.converged_reverse is False


@patch("sella.IRC")
def test_run_irc_reverse_only(MockIRC):
    MockIRC.return_value = _mock_irc_instance(converged=True)

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="reverse")
    assert MockIRC.call_count == 1
    assert result.converged_reverse is True
    assert result.converged_forward is False


@patch("sella.IRC")
def test_run_irc_both_converged(MockIRC):
    MockIRC.side_effect = [
        _mock_irc_instance(converged=True, nsteps=5),
        _mock_irc_instance(converged=True, nsteps=8),
    ]

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="both")
    assert result.converged_forward is True
    assert result.converged_reverse is True


@patch("sella.IRC")
def test_run_irc_endpoints_are_atoms(MockIRC):
    MockIRC.return_value = _mock_irc_instance(converged=True)

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="both")
    assert result.endpoint_forward is not None
    assert result.endpoint_reverse is not None
    assert len(result.endpoint_forward) == len(ts)
    assert len(result.endpoint_reverse) == len(ts)


@patch("sella.IRC")
def test_run_irc_n_steps(MockIRC):
    MockIRC.side_effect = [
        _mock_irc_instance(converged=True, nsteps=12),
        _mock_irc_instance(converged=True, nsteps=15),
    ]

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="both")
    assert result.n_steps_forward == 12
    assert result.n_steps_reverse == 15


@patch("sella.IRC")
def test_run_irc_exception_continues(MockIRC):
    """IRC exception should not propagate — converged flag stays False."""
    mock_instance = _mock_irc_instance()
    mock_instance.run.side_effect = RuntimeError("IRC diverged")
    mock_instance.nsteps = 0
    MockIRC.return_value = mock_instance

    ts = _make_ts()
    calc = _make_calc()

    result = run_irc(ts, calc, direction="both")
    assert result.converged_forward is False
    assert result.converged_reverse is False
    # Endpoints should still be set (atoms after failed IRC)
    assert result.endpoint_forward is not None
    assert result.endpoint_reverse is not None


@patch("sella.IRC")
def test_run_irc_kwargs_forwarded(MockIRC):
    """dx, irctol, eta, gamma should reach the IRC constructor."""
    MockIRC.return_value = _mock_irc_instance()

    ts = _make_ts()
    calc = _make_calc()

    run_irc(ts, calc, direction="forward", dx=0.05, irctol=0.005, eta=1e-3, gamma=0.2)

    _, kwargs = MockIRC.call_args
    assert kwargs["dx"] == 0.05
    assert kwargs["irctol"] == 0.005
    assert kwargs["eta"] == 1e-3
    assert kwargs["gamma"] == 0.2
