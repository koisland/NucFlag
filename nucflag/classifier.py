import gzip
import os
import shutil
import sys
from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pysam
import scipy.signal
from intervaltree import Interval, IntervalTree

from .constants import PLOT_DPI
from .io import get_coverage_by_base
from .misassembly import Misassembly
from .plot import plot_coverage
from .region import update_relative_ignored_regions


def peak_finder(
    data: np.ndarray,
    positions: np.ndarray,
    *,
    height: int,
    distance: int,
    width: int,
    group_distance: int = 5_000,
) -> IntervalTree:
    _, peak_info = scipy.signal.find_peaks(
        data, height=height, distance=distance, width=width
    )

    # Peaks are passed in sorted order.
    intervals = IntervalTree(
        Interval(positions[int(left_pos)], positions[int(right_pos)])
        for left_pos, right_pos in zip(peak_info["left_ips"], peak_info["right_ips"])
    )
    # Merge intervals.
    intervals.merge_overlaps(strict=False)

    merged_intervals = IntervalTree()
    sorted_intervals: Deque[Interval] = deque(sorted(intervals.iter()))

    # Then do second merge allowing greater distances.
    while len(sorted_intervals) >= 2:
        peak_1 = sorted_intervals.popleft()
        peak_2 = sorted_intervals.popleft()

        # | 1 | | 2 |
        dst_between = peak_2.begin - peak_1.end
        # Merge peaks if within a set distance.
        if dst_between < group_distance:
            new_peak = Interval(peak_1.begin, peak_2.end)
            sorted_intervals.appendleft(new_peak)
        else:
            merged_intervals.add(peak_1)
            # Exit when last interval checked.
            if len(sorted_intervals) == 0:
                merged_intervals.add(peak_2)
                break
            else:
                sorted_intervals.appendleft(peak_2)

    return merged_intervals


# https://stackoverflow.com/a/7353335
def consecutive(data, stepsize: int = 1):
    return np.split(data, np.where((np.diff(data) <= stepsize) == False)[0] + 1)  # noqa: E712


def filter_interval_expr(interval: Interval, *, col: str = "position") -> pl.Expr:
    return (pl.col(col) >= interval.begin) & (pl.col(col) <= interval.end)


def classify_misassemblies(
    df_cov: pl.DataFrame,
    *,
    config: dict[str, Any],
    ignored_regions: IntervalTree,
) -> tuple[pl.DataFrame, dict[Misassembly, IntervalTree]]:
    # Calculate std and mean for both most and second most freq read.
    # Remove gaps which would artificially lower mean.
    df_gapless = df_cov.filter(pl.col("first") != 0)
    mean_first, stdev_first = df_gapless["first"].mean(), df_gapless["first"].std()
    mean_second, stdev_second = df_gapless["second"].mean(), df_gapless["second"].std()
    del df_gapless

    # Calculate misjoin height threshold. Filters for some percent of the mean or static value.
    misjoin_height_thr = (
        mean_first * config["first"]["thr_misjoin_valley"]
        if isinstance(config["first"]["thr_misjoin_valley"], float)
        else config["first"]["thr_misjoin_valley"]
    )

    first_peak_height_thr = mean_first + (
        config["first"]["thr_peak_height_std_above"] * stdev_first
    )
    first_valley_height_thr = mean_first - (
        config["first"]["thr_valley_height_std_below"] * stdev_first
    )

    first_peak_coords = peak_finder(
        df_cov["first"],
        df_cov["position"],
        height=first_peak_height_thr,
        distance=config["first"]["thr_min_peak_horizontal_distance"],
        width=config["first"]["thr_min_peak_width"],
        group_distance=config["first"]["peak_group_distance"],
    )
    first_valley_coords = peak_finder(
        -df_cov["first"],
        df_cov["position"],
        # Account for when thr goes negative.
        height=-(
            misjoin_height_thr
            if first_valley_height_thr < 0
            else first_valley_height_thr
        ),
        distance=config["first"]["thr_min_valley_horizontal_distance"],
        width=config["first"]["thr_min_valley_width"],
        group_distance=config["first"]["valley_group_distance"],
    )

    # Remove secondary rows that don't meet minimal secondary coverage.
    second_thr = max(
        round(mean_first * config["second"]["thr_min_perc_first"]),
        round(
            mean_second + (config["second"]["thr_peak_height_std_above"] * stdev_second)
        ),
    )

    classified_second_outliers = set()
    df_second_outliers = df_cov.filter(pl.col("second") > second_thr)
    # Group consecutive positions allowing a maximum gap of stepsize.
    # Larger stepsize groups more positions.
    second_outliers_coords = []
    for grp in consecutive(
        df_second_outliers["position"], stepsize=config["second"]["group_distance"]
    ):
        if len(grp) < config["second"]["thr_min_group_size"]:
            continue
        second_outliers_coords.append(Interval(grp[0], grp[-1]))

    misassemblies: dict[Misassembly, IntervalTree] = {
        m: IntervalTree() for m in Misassembly
    }

    # Intersect intervals and classify collapses.
    for peak in first_peak_coords.iter():
        assert isinstance(peak, Interval)
        for second_outlier in second_outliers_coords:
            if peak.contains_interval(second_outlier):
                misassemblies[Misassembly.COLLAPSE_VAR].add(peak)
                classified_second_outliers.add(second_outlier)

        if peak not in misassemblies[Misassembly.COLLAPSE_VAR]:
            local_mean_collapse_first = (
                df_cov.filter(filter_interval_expr(peak))
                .median()
                .get_column("first")[0]
            )
            # If local median of suspected collapsed region is greater than thr, is a collapse.
            if local_mean_collapse_first > first_peak_height_thr:
                misassemblies[Misassembly.COLLAPSE].add(peak)

    # Classify gaps.
    df_gaps = df_cov.filter(pl.col("first") == 0)
    gaps = IntervalTree()
    for grp in consecutive(df_gaps["position"], stepsize=1):
        if len(grp) < 2:
            continue

        gap_len = grp[-1] - grp[0]

        if gap_len < config["gaps"]["thr_max_allowed_gap_size"]:
            continue

        gap = Interval(grp[0], grp[-1])
        gaps.add(gap)

    misassemblies[Misassembly.GAP] = gaps

    # Classify misjoins.
    for valley in first_valley_coords.iter():
        assert isinstance(valley, Interval)
        for second_outlier in second_outliers_coords:
            if valley.contains_interval(second_outlier):
                misassemblies[Misassembly.MISJOIN].add(valley)
                classified_second_outliers.add(second_outlier)

        # Otherwise, check if valley's median falls below threshold.
        # This means that while no overlapping secondary reads, low coverage means likely elsewhere.
        # Treat as a misjoin.
        if valley not in misassemblies[Misassembly.MISJOIN]:
            # Filter first to get general region.
            df_valley = df_cov.filter(filter_interval_expr(valley)).filter(
                pl.col("first") <= misjoin_height_thr
            )
            # Skip if fewer than 2 points found.
            if df_valley.shape[0] < config["first"]["thr_min_valley_width"]:
                continue

            # Get bounds of region and calculate median.
            # Avoid flagging if intersects gap region.
            df_valley = (
                df_valley
                if df_valley.shape[0] == 1
                else df_cov.filter(
                    filter_interval_expr(
                        Interval(
                            df_valley["position"].min(), df_valley["position"].max()
                        )
                    )
                )
            )
            if df_valley["first"].min() <= misjoin_height_thr and not misassemblies[
                Misassembly.GAP
            ].overlaps(valley.begin, valley.end):
                misassemblies[Misassembly.MISJOIN].add(valley)

    # Check remaining secondary regions not categorized.
    for second_outlier in second_outliers_coords:
        if second_outlier in classified_second_outliers:
            continue

        df_second_outlier = df_cov.filter(
            filter_interval_expr(second_outlier) & (pl.col("second") != 0)
        )
        df_second_outlier_het_ratio = df_second_outlier.median().with_columns(
            het_ratio=pl.col("second") / (pl.col("first") + pl.col("second"))
        )
        # Use het ratio to classify.
        # Low ratio consider collapse with var.
        if (
            df_second_outlier_het_ratio["het_ratio"][0]
            < config["second"]["thr_collapse_het_ratio"]
        ):
            misassemblies[Misassembly.COLLAPSE_VAR].add(second_outlier)
        else:
            misassemblies[Misassembly.MISJOIN].add(second_outlier)

    # Annotate df with misassembly.
    lf = df_cov.lazy().with_columns(status=pl.lit("Good"))

    if ignored_regions.is_empty():
        return lf.collect(), defaultdict()

    filtered_misassemblies: DefaultDict[Misassembly, IntervalTree] = defaultdict(
        IntervalTree
    )
    for mtype, regions in misassemblies.items():
        remove_regions = IntervalTree()
        for region in regions.iter():
            assert isinstance(region, Interval)
            # Remove ignored regions.
            if ignored_regions.overlaps(region.begin, region.end):
                remove_regions.add(region)
                continue

            lf = lf.with_columns(
                status=pl.when(filter_interval_expr(region))
                .then(pl.lit(mtype))
                .otherwise(pl.col("status"))
            )

        filtered_misassemblies[mtype] = regions.difference(remove_regions)

    # TODO: false dupes
    return lf.collect(), filtered_misassemblies


def classify_plot_assembly(
    infile: str,
    output_dir: str | None,
    output_cov_dir: str | None,
    threads: int,
    contig: str,
    start: int,
    end: int,
    config: dict[str, Any],
    overlay_regions: DefaultDict[int, IntervalTree],
    ignored_regions: IntervalTree,
) -> pl.DataFrame:
    contig_name = f"{contig}:{start}-{end}"
    sys.stderr.write(f"Reading in NucFreq from region: {contig_name}\n")

    try:
        bam = pysam.AlignmentFile(infile, threads=threads)
        cov_first_second = np.flip(
            np.sort(get_coverage_by_base(bam, contig, start, end), axis=1)
        ).transpose()
        df = pl.DataFrame(
            {
                "position": np.arange(start, end),
                "first": cov_first_second[0],
                "second": cov_first_second[1],
            }
        )
        del cov_first_second
    except ValueError:
        df = pl.read_csv(infile, separator="\t", has_header=True).select(
            "position", "first", "second"
        )

    # Update ignored regions if relative.
    ignored_regions = IntervalTree(
        update_relative_ignored_regions(ignored_regions, ctg_start=start, ctg_end=end)
    )

    df_group_labeled, misassemblies = classify_misassemblies(
        df,
        config=config,
        ignored_regions=ignored_regions,
    )

    if output_dir:
        _ = plot_coverage(df_group_labeled, misassemblies, contig, overlay_regions)

        sys.stderr.write(f"Plotting {contig_name}.\n")

        output_plot = os.path.join(output_dir, f"{contig_name}.png")
        plt.tight_layout()
        plt.savefig(output_plot, dpi=PLOT_DPI)

    if output_cov_dir:
        sys.stderr.write(f"Writing coverage bed file for {contig_name}.\n")

        output_bed = os.path.join(output_cov_dir, f"{contig_name}.bed")
        df_group_labeled.write_csv(output_bed, separator="\t")

        sys.stderr.write(f"Compressing coverage bed file for {contig_name}.\n")
        with open(output_bed, "rb") as f_in:
            with gzip.open(f"{output_bed}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(output_bed)

    df_misassemblies = pl.DataFrame(
        [
            (contig_name, interval.begin, interval.end, misasm)
            for misasm, intervals in misassemblies.items()
            for interval in intervals
        ],
        schema=["contig", "start", "stop", "misassembly"],
    )
    return df_misassemblies
