from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fabqc", description="Simulated visual QC plus deterministic RCA.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="Run attribution + PatchCore evaluation and write results/")
    ev.add_argument("--out", default="results")
    ev.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)
    if args.cmd == "eval":
        from fabqc.eval.run import run

        report = run(args.out, seed=args.seed)
        summary = {
            "attribution": report["attribution"]["summary"],
            "vision_auroc": report["vision"]["auroc"],
            "pipeline": report.get("pipeline"),
            "disclaimer": report["disclaimer"],
        }
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
