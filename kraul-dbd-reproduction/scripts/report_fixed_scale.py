from __future__ import annotations

import argparse

from kraul_dbd_reproduction.fixed_reporting import build_fixed_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild fixed-scale Kraul-GA tables and figures")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    print(build_fixed_report(args.experiment), flush=True)


if __name__ == "__main__":
    main()
