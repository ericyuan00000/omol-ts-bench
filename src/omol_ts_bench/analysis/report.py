"""Generate summary reports from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_path: str | Path) -> list[dict]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from benchmark results.

    Parameters
    ----------
    results
        List of per-reaction result dictionaries.

    Returns
    -------
    dict
        Summary statistics including success rates at each stage.
    """
    n_total = len(results)
    if n_total == 0:
        return {"n_total": 0}

    n_neb_converged = sum(1 for r in results if r.get("neb_converged"))
    n_sella_converged = sum(1 for r in results if r.get("sella_converged"))
    n_irc_converged = sum(
        1 for r in results
        if r.get("irc_converged_forward") and r.get("irc_converged_reverse")
    )
    n_validated = sum(1 for r in results if r.get("validation_both_match"))
    n_errors = sum(1 for r in results if r.get("error"))

    return {
        "n_total": n_total,
        "n_neb_converged": n_neb_converged,
        "n_sella_converged": n_sella_converged,
        "n_irc_both_converged": n_irc_converged,
        "n_validated": n_validated,
        "n_errors": n_errors,
        "rate_neb": n_neb_converged / n_total,
        "rate_sella": n_sella_converged / n_total,
        "rate_irc": n_irc_converged / n_total,
        "rate_validated": n_validated / n_total,
        "rate_error": n_errors / n_total,
    }


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    summary = compute_summary(results)

    print("=" * 60)
    print("OMol TS Benchmark Results Summary")
    print("=" * 60)
    print(f"{'Total reactions:':<35} {summary['n_total']}")
    print(f"{'NEB converged:':<35} {summary['n_neb_converged']} ({summary['rate_neb']:.1%})")
    print(f"{'Sella converged:':<35} {summary['n_sella_converged']} ({summary['rate_sella']:.1%})")
    print(f"{'IRC both directions converged:':<35} {summary['n_irc_both_converged']} ({summary['rate_irc']:.1%})")
    print(f"{'Validated (endpoints match):':<35} {summary['n_validated']} ({summary['rate_validated']:.1%})")
    print(f"{'Errors:':<35} {summary['n_errors']} ({summary['rate_error']:.1%})")
    print("=" * 60)

    # List failed reactions
    failed = [r for r in results if r.get("error")]
    if failed:
        print("\nFailed reactions:")
        for r in failed:
            print(f"  {r['reaction_id']}: {r['error']}")

    # List reactions that didn't validate
    not_validated = [
        r for r in results
        if r.get("validation_both_match") is False
    ]
    if not_validated:
        print(f"\nReactions with endpoint mismatch ({len(not_validated)}):")
        for r in not_validated:
            print(f"  {r['reaction_id']}")
