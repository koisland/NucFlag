DEF_CONFIG = {
    "first": dict(
        thr_min_peak_horizontal_distance=1,
        thr_min_peak_width=20,
        thr_min_valley_horizontal_distance=1,
        thr_min_valley_width=3,
        thr_peak_height_std_above=4,
        thr_valley_height_std_below=2,
        thr_misjoin_valley=3,
        valley_group_distance=500,
        peak_group_distance=500,
    ),
    "second": dict(
        thr_min_perc_first=0.1,
        thr_peak_height_std_above=3,
        group_distance=30_000,
        thr_min_group_size=3,
        thr_collapse_het_ratio=0.2,
    ),
    "gaps": dict(thr_max_allowed_gap_size=0),
}
