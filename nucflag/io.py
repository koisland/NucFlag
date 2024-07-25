import sys
import pysam

import numpy as np
import portion as pt

from .region import Action, ActionOpt, IgnoreOpt, Region
from typing import TextIO, Generator


def get_coverage_by_base(
    bam: pysam.AlignmentFile, contig: str, start: int, end: int
) -> np.ndarray:
    coverage = bam.count_coverage(
        contig, start=start, stop=end, read_callback="nofilter", quality_threshold=None
    )
    assert len(coverage) == 4
    return np.array(tuple(zip(*coverage)))


def read_bed_file(
    bed_file: TextIO,
) -> Generator[tuple[str, int, int, list[str]], None, None]:
    for line in bed_file.readlines():
        if line[0] == "#":
            continue
        chrm, start, end, *other = line.strip().split()
        yield (chrm, int(start), int(end), other)


def read_asm_regions(
    bamfile: str, input_regions: TextIO | None, *, threads: int = 4
) -> Generator[tuple[str, int, int], None, None]:
    if input_regions:
        sys.stderr.write(f"Reading in regions from {input_regions.name}.\n")

        yield from (
            (ctg, start, stop) for ctg, start, stop, *_ in read_bed_file(input_regions)
        )
    else:
        refs = {}
        with pysam.AlignmentFile(bamfile, threads=threads) as bam:
            sys.stderr.write(f"Reading entire {bam} because no bedfile was provided.\n")
            for read in bam.fetch(until_eof=True):
                ref = read.reference_name
                if ref not in refs:
                    refs[ref] = [2147483648, 0]

                start = read.reference_start
                end = read.reference_end
                if refs[ref][0] > start:
                    refs[ref][0] = start
                if refs[ref][1] < end:
                    refs[ref][1] = end

        for contig in refs:
            yield (contig, refs[contig][0], refs[contig][1])


def read_regions(bed_file: TextIO) -> Generator[Region, None, None]:
    for line in read_bed_file(bed_file):
        ctg, start, end, other = line
        try:
            desc = other[0]
        except IndexError:
            desc = None
        try:
            action_str = other[1]
            action_opt, _, action_desc = action_str.partition(":")
            action_opt = ActionOpt(action_opt)
            # TODO: Should be delimited by commas in case multiple needed.
            if action_opt == ActionOpt.IGNORE:
                action_desc = IgnoreOpt(action_desc)
            elif action_opt == ActionOpt.PLOT:
                action_desc = action_desc
            else:
                action_desc = None
            action = Action(action_opt, action_desc)
        except IndexError:
            action = None

        yield Region(name=ctg, region=pt.open(start, end), desc=desc, action=action)
