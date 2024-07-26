from typing import Any, DefaultDict

import matplotlib
import matplotlib.axes
import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from intervaltree import Interval, IntervalTree

from .constants import PLOT_HEIGHT, PLOT_HEIGHT_SCALE_FACTOR, PLOT_WIDTH, PLOT_YLIM
from .misassembly import Misassembly
from .region import ActionOpt, RegionInfo


def plot_coverage(
    df: pl.DataFrame,
    misassemblies: dict[Misassembly, IntervalTree],
    contig_name: str,
    overlay_regions: DefaultDict[int, IntervalTree],
) -> tuple[plt.Figure, Any]:
    region_bounds = Interval(df["position"].min(), df["position"].max())

    if overlay_regions:
        number_of_overlap_beds = len(overlay_regions.keys())
        fig, axs = plt.subplots(
            # nrows (1 for the coverage plot.)
            1 + len(overlay_regions.keys()),
            # ncols
            1,
            figsize=(
                PLOT_WIDTH,
                PLOT_HEIGHT + (PLOT_HEIGHT_SCALE_FACTOR * number_of_overlap_beds),
            ),
            sharex=True,
            gridspec_kw={
                "height_ratios": [*[0.5 for _ in range(number_of_overlap_beds)], 3]
            },
        )
        plt.subplots_adjust(hspace=0.6)

        # Last axis with coverage plt.
        ax: matplotlib.axes.Axes = axs[number_of_overlap_beds]
        # Add bed regions.
        for i, regions in overlay_regions.items():
            # Make axis as minimal as possible.
            bed_axs: matplotlib.axes.Axes = axs[i]
            bed_axs.get_xaxis().set_visible(False)
            bed_axs.get_yaxis().set_visible(False)
            bed_axs.set_frame_on(False)

            # Map uniq types to new color if none given.
            uniq_types = sorted({r.data.desc for r in regions.iter() if r.data.desc})
            cmap = dict(
                zip(
                    uniq_types,
                    (
                        c
                        for c in plt.colormaps.get_cmap("hsv")(
                            np.linspace(0, 1, len(uniq_types))
                        )
                    ),
                )
            )

            for row in regions.iter():
                assert isinstance(row, Interval)
                assert isinstance(row.data, RegionInfo)

                # Skip rows not within bounds of df.
                if not region_bounds.overlaps(row.begin, row.end):
                    continue

                if not row.data.action or (row.data.action.opt != ActionOpt.PLOT):
                    continue
                width = row.end - row.begin
                # Use color provided. Default to random generated ones otherwise.
                if row.data.action.desc:
                    color = row.data.action.desc
                elif row.data.desc:
                    color = cmap[row.data.desc]
                else:
                    raise ValueError(f"Region {row} has no description.")

                rect = ptch.Rectangle(
                    (row.begin, 0.7),
                    width,
                    0.5,
                    linewidth=1,
                    edgecolor=None,
                    facecolor=color,
                    alpha=0.75,
                    label=row.data.desc,
                )
                bed_axs.add_patch(rect)

            # Add legend.
            handles, labels = bed_axs.get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            bed_axs.legend(
                handles,
                labels,
                loc="lower center",
                borderaxespad=0,
                ncols=len(handles) // 2,
                fancybox=True,
                fontsize=12,
            )
    else:
        fig, axs = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        ax = axs

    (_,) = ax.plot(
        df["position"],
        df["first"],
        "o",
        color="black",
        markeredgewidth=0.0,
        markersize=2,
        label="Most Frequent Base",
    )
    (_,) = ax.plot(
        df["position"],
        df["second"],
        "o",
        color="red",
        markeredgewidth=0.0,
        markersize=2,
        label="Second Most Frequent Base",
    )
    # Add misassembly rect patches to highlight region.
    for misasm, misasm_regions in misassemblies.items():
        color = misasm.as_color()
        for misasm_region in misasm_regions.iter():
            assert isinstance(misasm_region, Interval)
            ax.axvspan(
                misasm_region.begin,
                misasm_region.end,
                color=color,
                alpha=0.4,
                label=misasm,
            )

    # Add legend. Deduplicate multiple labels.
    # https://stackoverflow.com/a/36189073
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=len(labels),
        borderaxespad=0,
        fancybox=True,
    )

    maxval = df["position"].max()
    minval = df["position"].min()
    subval = 0

    title = "{}:{}-{}\n".format(contig_name, minval, maxval)
    plt.suptitle(title, fontweight="bold")

    if maxval < 1_000_000:
        xlabels = [format((label - subval), ",.0f") for label in ax.get_xticks()]
        lab = "bp"
    elif maxval < 10_000_000:
        xlabels = [format((label - subval) / 1000, ",.1f") for label in ax.get_xticks()]
        lab = "kbp"
    else:
        xlabels = [format((label - subval) / 1000, ",.1f") for label in ax.get_xticks()]
        lab = "kbp"

    ax.set_ylim(0, PLOT_YLIM)
    ax.set_xlabel("Assembly position ({})".format(lab), fontweight="bold")
    ax.set_ylabel("Sequence read depth", fontweight="bold")
    ax.set_xticklabels(xlabels)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    fig.tight_layout()

    return fig, axs
