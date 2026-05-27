"""CLI entry point: python -m sigma_diff file_a.py file_b.py [--format=text|json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sigma_diff",
        description="sigma-diff: behavioral diff engine for code files",
    )
    parser.add_argument("file_a", nargs="?", help="First (old) Python file")
    parser.add_argument("file_b", nargs="?", help="Second (new) Python file")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--ryzanstein-url",
        default=None,
        help="Optional Ryzanstein embedding server URL",
    )

    args = parser.parse_args()

    if not args.file_a or not args.file_b:
        parser.print_help()
        sys.exit(0)

    path_a = Path(args.file_a)
    path_b = Path(args.file_b)

    if not path_a.exists():
        print(f"Error: {path_a} does not exist", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: {path_b} does not exist", file=sys.stderr)
        sys.exit(1)

    code_a = path_a.read_text(encoding="utf-8")
    code_b = path_b.read_text(encoding="utf-8")

    from sigma_diff import SigmaDiff

    engine = SigmaDiff(ryzanstein_url=args.ryzanstein_url)
    diff_result, score, summary = engine.analyze(code_a, code_b)

    if args.format == "json":
        out = {
            "diff": {
                "additions": diff_result.additions,
                "deletions": diff_result.deletions,
                "moved_blocks": [list(m) for m in diff_result.moved_blocks],
                "structural_similarity": diff_result.structural_similarity,
                "node_changes": [
                    {
                        "change_type": nc.change_type,
                        "node_name": nc.node_name,
                        "old_value": nc.old_value,
                        "new_value": nc.new_value,
                    }
                    for nc in diff_result.node_changes
                ],
            },
            "score": {
                "structural": score.structural,
                "semantic": score.semantic,
                "combined": score.combined,
                "risk_level": score.risk_level,
            },
            "summary": summary,
        }
        print(json.dumps(out, indent=2))
    else:
        print(summary)


if __name__ == "__main__":
    main()
