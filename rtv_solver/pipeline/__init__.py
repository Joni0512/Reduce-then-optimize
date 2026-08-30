from .feat_builder import FeatureBuilder, VehicleFeatures, StateFeatures, TripFeatures, TripCostFeatures
# 2026-08-05: aliased import for the v2 feature-builder comparison - feat_builder_new.py
# defines a class of the same name "FeatureBuilder", so it must be imported under an
# alias here rather than reassigning the name above (which would silently make every
# existing `from rtv_solver.pipeline import FeatureBuilder` caller get v2 instead of v1).
from .feat_builder import FeatureBuilder as FeatureBuilderV1
from .feat_builder_new import FeatureBuilder as FeatureBuilderV2
from .co_base import CO
from .co_tripCostMinimization import CO_TripCostMinimization
from .co_scoreMaximization import CO_ScoreMaximization
from .co_rebalancing import CO_RebalancingCoverage
from .model_simpleScoring import ScoringMLP
#neue datei
from rtv_solver.pipeline.request_graph_feature_builder import (
    RequestGraphFeatureBuilder,
)
# loss and oracle imports come after co_scoreMaximization to prevent circular imports
# (map_oracle imports CO_ScoreMaximization directly by module path)
from .loss_FYscoring import FenchelYoungLoss
from .map_oracle import make_map_oracle
from rtv_solver.pipeline.imitation_handler import ImitationHandler, TYPE_BEST_ORDERED_MATCH, TYPE_BEST_UNORDERED_MATCH


def select_feature_builder_class(config):
    """
    2026-08-30: extracted out of build_feature_builder so training_loop.py's
    TrainingLoop.__init__ (which needs the class + its FEATURE_SIZE for model
    construction, before any payload exists to build an instance) can apply
    the same --enable_pickup_slack_feature override, instead of picking
    FeatureBuilderV1/V2 directly and silently skipping it. That mismatch (model
    sized for the un-overridden FEATURE_SIZE, actual feature dicts built via
    the overridden one) is exactly what caused "Expected 85 input features,
    received 84" the first time this override was tested end-to-end.
    """
    builder_cls = FeatureBuilderV2 if config.FEATURE_BUILDER_VERSION == "v2" else FeatureBuilderV1
    override = getattr(config, "ENABLE_PICKUP_SLACK_FEATURE", None)
    if override is not None and override != builder_cls.ENABLE_PICKUP_SLACK_FEATURE:
        # FEATURE_SIZE is a plain int computed once at class-body execution
        # time from ENABLE_PICKUP_SLACK_FEATURE's hardcoded default - flipping
        # the flag alone would leave model construction
        # (type(fb).FEATURE_SIZE, see coaml_pipeline.py/training_loop.py) out of
        # sync with the actual per-row feature dict. Mutate both together,
        # process-wide (same single-config-per-run assumption ENABLE_COMPETITION_FEATURES
        # already relies on).
        builder_cls.ENABLE_PICKUP_SLACK_FEATURE = override
        builder_cls.FEATURE_SIZE = builder_cls.FEATURE_SIZE + (
            builder_cls._PICKUP_SLACK_FEATURE_SIZE if override else -builder_cls._PICKUP_SLACK_FEATURE_SIZE
        )
    return builder_cls


def build_feature_builder(complete_payload: dict, config) -> FeatureBuilder:
    """
    2026-08-05: single switch point for the v1/v2 feature-builder comparison
    (config.FEATURE_BUILDER_VERSION, default "v1" = unchanged feat_builder.py
    behavior). Callers that need the active FEATURE_SIZE for model construction
    (coaml_pipeline.py, training_loop.py) should read it off the returned
    instance's class (type(fb).FEATURE_SIZE), not off the FeatureBuilder name
    directly, since that name always refers to v1.
    """
    builder_cls = select_feature_builder_class(config)
    return builder_cls(complete_payload, config)


__all__ = [
    "FeatureBuilder",
    "FeatureBuilderV1",
    "FeatureBuilderV2",
    "build_feature_builder",
    "select_feature_builder_class",
    "RequestGraphFeatureBuilder",
    "VehicleFeatures",
    "StateFeatures",
    "TripFeatures",
    "TripCostFeatures",
    "CO", 
    "CO_TripCostMinimization", 
    "CO_ScoreMaximization",  
    "CO_RebalancingCoverage",
    "ScoringMLP",
    "FenchelYoungLoss",
    "make_map_oracle",
    "ImitationHandler",
    "TYPE_BEST_ORDERED_MATCH",
    "TYPE_BEST_UNORDERED_MATCH"
]
