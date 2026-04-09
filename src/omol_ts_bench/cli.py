"""Command-line interface for omol-ts-bench."""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OMol TS Benchmark: Transition state optimization benchmark for organometallic reactions",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the benchmark pipeline")
    run_parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to data directory containing xyz/ subdirectories",
    )
    run_parser.add_argument(
        "--tier", type=str, default=None, choices=["smoke", "standard", "full"],
        help="Reaction tier to run (default: all)",
    )
    run_parser.add_argument(
        "--registry", type=str, default=None,
        help="Path to reactions.yaml registry file",
    )
    run_parser.add_argument(
        "--method", type=str, default="neb", choices=["neb", "gsm"],
        help="Double-ended TS search method",
    )
    run_parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (simple workflow only)",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--workflow", type=str, default="simple", choices=["simple", "quacc"],
        help="Workflow engine to use",
    )
    run_parser.add_argument(
        "--calc", type=str, default="omol_ts_bench.core.calculator.get_uma_calculator",
        help="Dotted path to calculator factory function",
    )
    run_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file for NEB/Sella/IRC parameters",
    )

    # --- report ---
    report_parser = subparsers.add_parser("report", help="Generate summary report from results")
    report_parser.add_argument(
        "results_path", type=str,
        help="Path to results.json file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run":
        _handle_run(args)
    elif args.command == "report":
        _handle_report(args)
    else:
        parser.print_help()
        sys.exit(1)


def _handle_run(args):
    """Handle the 'run' command."""
    import yaml

    from omol_ts_bench.data.loader import load_reactions_from_directory

    # Load config
    neb_kwargs = {}
    sella_kwargs = {}
    irc_kwargs = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        neb_kwargs = config.get("neb", {})
        sella_kwargs = config.get("sella", {})
        irc_kwargs = config.get("irc", {})

    # Load reactions
    reactions = load_reactions_from_directory(
        data_dir=args.data_dir,
        tier=args.tier,
        registry_path=args.registry,
    )
    logging.info("Loaded %d reactions (tier=%s)", len(reactions), args.tier)

    if not reactions:
        logging.warning("No reactions found in %s", args.data_dir)
        return

    if args.workflow == "simple":
        from omol_ts_bench.workflows.simple import run_benchmark

        results = run_benchmark(
            reactions=reactions,
            calc_factory=args.calc,
            double_ended_method=args.method,
            n_workers=args.n_workers,
            output_dir=args.output_dir,
            neb_kwargs=neb_kwargs,
            sella_kwargs=sella_kwargs,
            irc_kwargs=irc_kwargs,
        )

        from omol_ts_bench.analysis.report import print_summary
        print_summary(results)

    elif args.workflow == "quacc":
        import importlib
        module_path, func_name = args.calc.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        calc_factory = getattr(mod, func_name)

        from omol_ts_bench.workflows.quacc import run_benchmark_quacc

        futures = run_benchmark_quacc(
            reactions=reactions,
            calc_factory=calc_factory,
            double_ended_method=args.method,
            neb_kwargs=neb_kwargs,
            sella_kwargs=sella_kwargs,
            irc_kwargs=irc_kwargs,
        )
        logging.info("Submitted %d jobs via QuAcc", len(futures))


def _handle_report(args):
    """Handle the 'report' command."""
    from omol_ts_bench.analysis.report import load_results, print_summary

    results = load_results(args.results_path)
    print_summary(results)


if __name__ == "__main__":
    main()
