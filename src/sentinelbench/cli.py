from __future__ import annotations

import argparse
import json

from .reporting import project_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="posttrain-lab", description="Inspect the post-training lab")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the JSON report for readability.",
    )
    arguments = parser.parse_args()
    print(json.dumps(project_report(), indent=2 if arguments.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()