"""
Utility script to transform state data files according to custom rules.

Rules implemented:
1. Starting from a specified line number (inclusive, 1-indexed), adjust x/y/z:
   - x += 1
   - y -= 0.5
   - z -= 0.1
   (line counting includes the header line, which is typically line 1)
2. After the above conditional adjustment, add 1.3 to the z value for all data rows.
3. Write the transformed content to a new file, preserving tab-separated formatting
   with six decimal places for numeric fields.

Usage:
    python adjust_state_data.py --input hnn/state_data.txt \
        --output hnn/state_data_adjusted.txt --threshold 187
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, List, Optional
import sys


DECIMAL_PLACES = Decimal("0.000001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjust x/y/z columns of a state_data.txt-style file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the source txt file. If omitted the script will search for candidates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the transformed txt file. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="1-indexed line number (including header) from which x/y/z adjustments apply. If omitted you will be prompted.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without interactive prompts: pick the first candidate and use defaults.",
    )
    return parser.parse_args()


def quantize(value: Decimal) -> str:
    return format(value.quantize(DECIMAL_PLACES, rounding=ROUND_HALF_UP), "f")


def transform_lines(
    lines: Iterable[str], threshold: int
) -> List[str]:  # pragma: no cover - straightforward iteration
    line_iter = iter(lines)
    header = next(line_iter, None)
    if header is None:
        raise ValueError("Input file is empty.")

    output_lines: List[str] = [header.rstrip("\n")]
    for idx, raw_line in enumerate(line_iter, start=2):
        line = raw_line.rstrip("\n")
        if not line.strip():
            output_lines.append(line)
            continue

        parts = line.split("\t")
        if len(parts) != 10:
            output_lines.append(line)
            continue

        values = [Decimal(part) for part in parts]
        if idx <= threshold:
            values[1] -= Decimal("0.0")
            values[2] -= Decimal("0.0")
            values[3] += Decimal("0.0")
            # values[6] += Decimal(f"{math.radians(60)}")
        else:
            values[1] += Decimal("0.9")
            values[2] -= Decimal("0.0")
            values[3] += Decimal("0.9")
        formatted = "\t".join(quantize(v) for v in values)
        output_lines.append(formatted)

    return output_lines


DEFAULT_THRESHOLD = 187


def find_state_files(root: Path) -> List[Path]:
    """Recursively find files named `state_data.txt` under `root`.

    Returns a sorted list of Path objects.
    """
    return sorted(root.rglob("state_data.txt"))


def choose_candidate(candidates: List[Path], non_interactive: bool) -> Path:
    if not candidates:
        raise SystemExit("No 'state_data.txt' files found under the current directory.")
    if non_interactive or len(candidates) == 1:
        return candidates[0]

    cwd = Path.cwd()
    print("Found multiple candidate files:")
    for i, p in enumerate(candidates, start=1):
        try:
            rel = p.relative_to(cwd)
        except Exception:
            rel = p
        print(f"  {i}) {rel}")

    while True:
        s = input(f"Select file by number [1]: ").strip()
        if not s:
            return candidates[0]
        try:
            idx = int(s)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def prompt_threshold(default: int, non_interactive: bool) -> int:
    if non_interactive:
        return default
    while True:
        s = input(f"Threshold (1-indexed, header included) [{default}]: ").strip()
        if not s:
            return default
        try:
            val = int(s)
            if val >= 1:
                return val
        except ValueError:
            pass
        print("Please enter a positive integer for the threshold.")


def main() -> None:
    args = parse_args()

    # determine input file
    if args.input is None:
        candidates = find_state_files(Path.cwd())
        chosen = choose_candidate(candidates, args.non_interactive)
    else:
        chosen = args.input

    if not chosen.exists():
        raise SystemExit(f"Input file does not exist: {chosen}")

    # determine output path
    if args.output is None:
        out_path = chosen.parent / "state_data_adjusted.txt"
    else:
        out_path = args.output

    # determine threshold (prompt each interactive run)
    initial_threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    threshold = prompt_threshold(initial_threshold, args.non_interactive)

    source_lines = chosen.read_text(encoding="utf-8").splitlines()
    transformed = transform_lines(source_lines, threshold)
    out_path.write_text("\n".join(transformed) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


