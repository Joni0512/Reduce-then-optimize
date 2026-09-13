"""
2026-09-13: stratified 38/9/9 train/val/test split over all 56 Li&Lim
instances (lc1:9, lc2:8, lr1:12, lr2:11, lrc1:8, lrc2:8), evenly spread over
each category's numeric range rather than random or "last N" - see chat.
Used by the new SIL / SRL(local) / SRL(local_positive) training loop and its
wandb sweeps, replacing the older ad-hoc 12/12-instance splits scattered
across individual run_srl_*_12instances.py scripts.

OVERFIT_CHECK_INSTANCES: 9 instances FROM TRAIN_INSTANCES, tracked in
validation-time plots to compare train-fit vs. val-fit over epochs. NEVER
used for best-checkpoint model selection - only VAL_INSTANCES decides that.
2026-09-13: stratified like the main split (proportional to TRAIN_INSTANCES'
own category composition: lc1x2, lc2x1, lr1x2, lr2x2, lrc1x1, lrc2x1 = 9) -
an earlier "just take the first 9" version was all lc1/lc2, missing lr/lrc
and class 2 entirely (see chat).
"""

TRAIN_INSTANCES = [
    "lc101", "lc102", "lc104", "lc105", "lc106", "lc107", "lc109",
    "lc201", "lc203", "lc204", "lc206", "lc207",
    "lr101", "lr102", "lr104", "lr105", "lr106", "lr108", "lr109", "lr110", "lr112",
    "lr201", "lr203", "lr204", "lr206", "lr207", "lr209", "lr210",
    "lrc101", "lrc103", "lrc104", "lrc106", "lrc107",
    "lrc201", "lrc203", "lrc204", "lrc206", "lrc207",
]

VAL_INSTANCES = [
    "lc103", "lc202",
    "lr103", "lr107", "lr202", "lr208",
    "lrc102", "lrc202", "lrc205",
]

TEST_INSTANCES = [
    "lc108", "lc205", "lc208",
    "lr111", "lr205", "lr211",
    "lrc105", "lrc108", "lrc208",
]

OVERFIT_CHECK_INSTANCES = [
    "lc102", "lc107", "lc204",
    "lr104", "lr109", "lr203", "lr209",
    "lrc104", "lrc204",
]

assert len(TRAIN_INSTANCES) == 38
assert len(VAL_INSTANCES) == 9
assert len(TEST_INSTANCES) == 9
assert not (set(TRAIN_INSTANCES) & set(VAL_INSTANCES) & set(TEST_INSTANCES))
assert set(OVERFIT_CHECK_INSTANCES) <= set(TRAIN_INSTANCES)
