#!/usr/bin/env python
import argparse
import io
import multiprocessing as mp
import os
import pprint
import sys
import warnings
from collections import defaultdict
from typing import DefaultDict

import matplotlib
import polars as pl
import tomllib
from intervaltree import IntervalTree

from .classifier import classify_plot_assembly
from .config import DEF_CONFIG
from .constants import (
    PLOT_FONT_SIZE,
)
from .io import (
    read_asm_regions,
    read_ignored_regions,
    read_overlay_regions,
)
from .region import RegionStatus

matplotlib.use("agg")
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use per-base read coverage to classify/plot misassemblies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--infile",
        required=True,
        help="Input bam file or per-base coverage tsv file with 3-columns (position, first, second). If a bam file is provided, it must be indexed.",
    )
    parser.add_argument(
        "-b",
        "--input_regions",
        default=None,
        type=argparse.FileType("rt"),
        help="Bed file with regions to check.",
    )
    parser.add_argument(
        "-d",
        "--output_plot_dir",
        default=None,
        help="Output plot dir.",
    )
    parser.add_argument(
        "--output_cov_dir",
        default=None,
        help="Output coverage dir. Generates gzipped coverage bed files per region.",
    )
    parser.add_argument(
        "-o",
        "--output_misasm",
        default=sys.stdout,
        type=argparse.FileType("wt"),
        help="Output bed file with misassembled regions.",
    )
    parser.add_argument(
        "-s",
        "--output_status",
        default=None,
        type=argparse.FileType("wt"),
        help="Bed file with status of contigs. With format: contig\tstart\tend\tmisassembled|good",
    )
    parser.add_argument(
        "-t", "--threads", default=4, type=int, help="Threads for reading bam file."
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=4,
        type=int,
        help="Processes for classifying/plotting.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEF_CONFIG,
        type=argparse.FileType("rb"),
        help="Additional threshold/params as toml file.",
    )
    parser.add_argument(
        "--ignore_regions",
        default=None,
        type=argparse.FileType("rt"),
        help="Bed file with regions to ignore. With format: contig|all\tstart\tend\tabsolute|relative",
    )
    parser.add_argument(
        "--overlay_regions",
        nargs="*",
        type=argparse.FileType("rt"),
        help="Overlay additional regions as 4-column bedfile alongside coverage plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_plot_dir:
        os.makedirs(args.output_plot_dir, exist_ok=True)
    if args.output_cov_dir:
        os.makedirs(args.output_cov_dir, exist_ok=True)

    if isinstance(args.config, io.IOBase):
        config = tomllib.load(args.config)
    else:
        config = args.config

    sys.stderr.write(f"Using config:\n{pprint.pformat(config)}.\n")

    # Read regions in bam or from input regions.
    regions = list(
        read_asm_regions(args.infile, args.input_regions, threads=args.threads)
    )
    sys.stderr.write(f"Loaded {len(regions)} region(s).\n")

    # Load ignored regions.
    if args.ignore_regions:
        ignored_regions: DefaultDict[str, IntervalTree] = read_ignored_regions(
            args.ignore_regions
        )
        total_ignored_positions = sum(len(v) for _, v in ignored_regions.items())
        total_ignored_regions = len(
            regions if "all" in ignored_regions else ignored_regions
        )
        sys.stderr.write(
            f"Ignoring {total_ignored_positions} position(s) from {total_ignored_regions} region(s).\n"
        )
    else:
        ignored_regions = defaultdict(IntervalTree)

    # Load additional regions to overlay.
    if args.overlay_regions:
        overlay_regions = read_overlay_regions(args.overlay_regions)
        sys.stderr.write(f"Overlapping {len(overlay_regions)} bedfile(s).\n")
    else:
        overlay_regions = defaultdict(lambda: defaultdict(IntervalTree))

    # Set text size
    matplotlib.rcParams.update({"font.size": PLOT_FONT_SIZE})

    # results = []
    # for region in regions:
    #     results.append(
    #         classify_plot_assembly(
    #             args.infile,
    #             args.output_plot_dir,
    #             args.output_cov_dir,
    #             args.threads,
    #             *region,
    #             config,
    #             overlay_regions.get(region[0], defaultdict(IntervalTree)),
    #             (
    #                 ignored_regions["all"]
    #                 if "all" in ignored_regions
    #                 else ignored_regions.get(region[0], IntervalTree())
    #             ),
    #         )
    #     )

    with mp.Pool(processes=args.processes) as pool:
        results = pool.starmap(
            classify_plot_assembly,
            [
                (
                    args.infile,
                    args.output_plot_dir,
                    args.output_cov_dir,
                    args.threads,
                    *region,
                    config,
                    overlay_regions.get(region[0], defaultdict(IntervalTree)),
                    # "all" takes precedence.
                    (
                        ignored_regions["all"]
                        if "all" in ignored_regions
                        else ignored_regions.get(region[0], IntervalTree())
                    ),
                )
                for region in regions
            ],
        )

    # Save misassemblies to output bed
    dfs_misasm = []
    region_status = []
    for region_info, df_misasm in zip(regions, results):
        if not df_misasm.is_empty():
            region_status.append((*region_info, RegionStatus.MISASSEMBLED))
            dfs_misasm.append(df_misasm)
        else:
            region_status.append((*region_info, RegionStatus.GOOD))

    try:
        df_all_misasm: pl.DataFrame = pl.concat(dfs_misasm)
    except ValueError:
        df_all_misasm = pl.DataFrame(schema=["contig", "start", "end"])

    df_all_misasm.sort(by=["contig", "start"]).write_csv(
        file=args.output_misasm, include_header=False, separator="\t"
    )

    if args.output_status:
        df_asm_status = pl.DataFrame(
            region_status, schema=["contig", "start", "end", "status"]
        )
        df_asm_status.sort(by=["contig", "start"]).write_csv(
            args.output_status, include_header=False, separator="\t"
        )


if __name__ == "__main__":
    raise SystemExit(main())
