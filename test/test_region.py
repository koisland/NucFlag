import pytest
from intervaltree import Interval, IntervalTree

from nucflag.region import (
    Action,
    ActionOpt,
    IgnoreOpt,
    RegionInfo,
    update_relative_ignored_regions,
)


def test_update_relative_region():
    # lower bound must be smaller than upper bound
    with pytest.raises(ValueError):
        itree = IntervalTree(
            [
                Interval(0, -250),
                RegionInfo("", None, Action(ActionOpt.IGNORE, desc=IgnoreOpt.RELATIVE)),
            ]
        )
        update_relative_ignored_regions(itree, ctg_start=0, ctg_end=100)

    with pytest.raises(ValueError):
        itree = IntervalTree(
            [
                Interval(250, 0),
                RegionInfo("", None, Action(ActionOpt.IGNORE, desc=IgnoreOpt.RELATIVE)),
            ]
        )
        update_relative_ignored_regions(itree, ctg_start=0, ctg_end=100)
