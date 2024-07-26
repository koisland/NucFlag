from enum import StrEnum, auto
from typing import Any, Generator, NamedTuple

import numpy as np
from intervaltree import Interval, IntervalTree


class RegionStatus(StrEnum):
    MISASSEMBLED = auto()
    GOOD = auto()


class IgnoreOpt(StrEnum):
    ABSOLUTE = auto()
    RELATIVE = auto()


class ActionOpt(StrEnum):
    IGNORE = auto()
    PLOT = auto()


class Action(NamedTuple):
    opt: ActionOpt
    desc: Any | None


class RegionInfo(NamedTuple):
    name: str
    desc: str | None
    action: Action | None


def update_relative_ignored_regions(
    ignored_regions: IntervalTree, *, ctg_start: int, ctg_end: int
) -> Generator[Interval, None, None]:
    for region in ignored_regions.iter():
        assert isinstance(region, Interval)
        assert isinstance(region.data, RegionInfo)

        if (
            not region.data.action
            or (region.data.action and region.data.action.opt != ActionOpt.IGNORE)
            or (region.data.action and region.data.action.desc != IgnoreOpt.RELATIVE)
        ):
            continue

        if region.begin > region.end:
            raise ValueError(
                f"Region lower bound cannot be larger than upper bound. ({region})"
            )
        if region.begin < 0:
            rel_start = ctg_end
        else:
            rel_start = ctg_start

        lower = rel_start + region.begin
        upper = rel_start + region.end

        yield Interval(max(lower, 0), np.clip(upper, 0, ctg_end), region.data)
