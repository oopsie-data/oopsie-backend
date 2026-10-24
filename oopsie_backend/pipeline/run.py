"""Run pipeline stages with Hamilton, recomputing only what changed.

Usage:
    uv run python -m oopsie_backend.pipeline.run [TARGET ...] --config configs/pipeline.yaml
    uv run python -m oopsie_backend.pipeline.run --list
    uv run python -m oopsie_backend.pipeline.run qa --graph pipeline.png

A target is a name from ``pipeline/stages.py``'s ``TARGETS`` (default ``nightly``)
or a stage name. Hamilton computes the requested stages and everything upstream;
a stage whose code and upstream fingerprints are unchanged since its last run is
taken from the cache in ``cache_dir``. Independent stages run in parallel.
The config file holds every input a stage may need (``configs/pipeline.yaml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from hamilton import ad_hoc_utils, driver
from hamilton.plugins.h_threadpool import FutureAdapter

from oopsie_backend.pipeline.stages import STAGES, TARGETS

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def build_driver(cache_dir: Path | None, parallel: bool = True) -> driver.Driver:
    module = ad_hoc_utils.create_temporary_module(*STAGES.values(), module_name="oopsie_pipeline")
    builder = driver.Builder().with_modules(module)
    if cache_dir is not None:
        builder = builder.with_cache(path=str(cache_dir))
    if parallel:
        builder = builder.with_adapters(FutureAdapter())
    return builder.build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", default=["nightly"], metavar="TARGET")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config input; repeatable"
    )
    parser.add_argument("--list", action="store_true", help="List targets and stages")
    parser.add_argument("--graph", type=Path, help="Draw the stages the targets need (needs graphviz)")
    parser.add_argument("--no-cache", action="store_true", help="Run every requested stage")
    parser.add_argument("--serial", action="store_true", help="Run stages one at a time")
    parser.add_argument("--explain", action="store_true", help="Print which stages ran and which came from the cache")
    args = parser.parse_args(argv)

    if args.list:
        for name, stages in TARGETS.items():
            print(f"{name:<11} {', '.join(stages)}")
        print("\nstages: " + ", ".join(sorted(STAGES)))
        return 0

    unknown = [t for t in args.targets if t not in TARGETS and t not in STAGES]
    if unknown:
        print(f"error: unknown targets {unknown}; see --list", file=sys.stderr)
        return 2
    final = sorted({s for t in args.targets for s in TARGETS.get(t, [t])})
    config = yaml.safe_load(args.config.read_text())
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in config:
            print(f"error: --set expects KEY=VALUE with KEY in {args.config}, got {item!r}", file=sys.stderr)
            return 2
        config[key] = yaml.safe_load(value)
    cache_dir = Path(config.pop("cache_dir"))
    if args.no_cache:
        cache_dir = None
    dr = build_driver(cache_dir, parallel=not args.serial)
    if args.graph:
        dr.visualize_execution(final, str(args.graph.with_suffix("")), inputs=config, render_kwargs={"format": args.graph.suffix[1:] or "png"})
        print(f"wrote {args.graph}", file=sys.stderr)
        return 0
    results = dr.execute(final, inputs=config)
    for name in final:
        print(f"{name}: {results[name]}", file=sys.stderr)
    if args.explain and cache_dir is not None:
        for node, events in sorted(dr.cache.logs(run_id=dr.cache.last_run_id, level="info").items()):
            ran = any(getattr(e, "event_type", None) and e.event_type.name == "EXECUTE_NODE" for e in events)
            print(f"  {node:<20} {'ran' if ran else 'cached'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
