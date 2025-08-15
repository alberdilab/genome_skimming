#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count reads per sample from FASTQ files and write a two-column TSV: sample, depth.

Defaults:
- Expect R1 files only for paired-end (depth = number of read pairs).
- Ignores R2 files unless --include-r2 is given.
- Works with .fq or .fastq (optionally .gz as well).

Examples
--------
# Paired-end (R1 files only):
python reads_per_sample.py --out results/qc/read_depth.tsv data/fastq/*_R1.fq

# Single-end:
python reads_per_sample.py --out results/qc/read_depth.tsv data/fastq/*.fq

# If you *really* want to include R2 files in the count (sums all reads):
python reads_per_sample.py --include-r2 --out results/qc/read_depth.tsv data/fastq/*.fq
"""

import argparse
import re
from pathlib import Path
import gzip

def parse_args():
    ap = argparse.ArgumentParser(description="Produce two-column file: sample, depth.")
    ap.add_argument("--out", required=True, help="Output TSV path (sample\\tdepth).")
    ap.add_argument("--include-r2", action="store_true",
                    help="Include files that look like R2 mates (by default they are ignored).")
    ap.add_argument("fastq", nargs="+", help="Input FASTQ files (.fq/.fastq; gz or plain).")
    return ap.parse_args()

# --- helpers ---

_R1_TOKEN = re.compile(r'(?i)(?:^|[_\.\-])R?1(?:[_\.\-]|$)')
_R2_TOKEN = re.compile(r'(?i)(?:^|[_\.\-])R?2(?:[_\.\-]|$)')

def looks_like_r2(name: str) -> bool:
    return bool(_R2_TOKEN.search(name))

def infer_sample_name(p: Path) -> str:
    """
    Infer sample name by stripping a trailing R1/R2 (or 1/2) token before the extension.
    Examples:
      SAMPLE_A_R1.fq      -> SAMPLE_A
      SAMPLE-A.1.fastq    -> SAMPLE-A
      dog.R2.fq           -> dog
      run42_sampleX.fq    -> run42_sampleX
    """
    base = p.name
    # remove extension(s)
    stem = re.sub(r'\.(fastq|fq)(\.gz)?$', '', base, flags=re.IGNORECASE)

    # strip a single trailing mate token
    stem2 = re.sub(r'(?i)(?:[_\.\-])R?[12]$', '', stem)
    return stem2

def count_lines_fast(path: Path) -> int:
    """Count newline bytes quickly; return line count."""
    total = 0
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rb")
    else:
        fh = open(path, "rb")
    with fh:
        while True:
            chunk = fh.read(8 << 20)  # 8 MiB
            if not chunk:
                break
            total += chunk.count(b"\n")
    return total

def count_reads_fastq(path: Path) -> int:
    return count_lines_fast(path) // 4

# --- main ---

def main():
    args = parse_args()
    out_path = Path(args.out)
    counts = {}

    for f in sorted(set(args.fastq)):
        fp = Path(f)
        name = fp.name

        # Skip R2 unless user asked to include
        if (not args.include_r2) and looks_like_r2(name):
            continue

        sample = infer_sample_name(fp)
        reads = count_reads_fastq(fp)
        counts[sample] = counts.get(sample, 0) + reads

    # Write TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as oh:
        oh.write("sample\tdepth\n")
        for sample in sorted(counts):
            oh.write(f"{sample}\t{counts[sample]}\n")

    print(f"Wrote {out_path} with {len(counts)} samples.")

if __name__ == "__main__":
    main()
