"""
Import test: verify that all modules can be imported and basic functions run without error.

Run from project root:
python test_modules.py --data data/penguins.csv --report_dir reports
"""

from pathlib import Path
import argparse

from src import ensure_dirs, read_data, basic_profile, split_columns


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Test Build1 src module imports + basic functions."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Folder for outputs."
    )
    args = parser.parse_args()

    print("Data file:", args.data)
    print("Report folder:", args.report_dir)

    # Run a small smoke test of the imported functions
    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    df = read_data(Path(args.data))

    profile = basic_profile(df)
    print(profile)

    numeric_cols, cat_cols = split_columns(df)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", cat_cols)


if __name__ == "__main__":
    main()
