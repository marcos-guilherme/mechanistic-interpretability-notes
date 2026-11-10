"""Paper-faithful data and metrics for the IOI replication track."""

from ioi_replication.paper.dataset import (
    ABBA_TEMPLATES,
    BABA_TEMPLATES,
    HISTORICAL_NAMES,
    HISTORICAL_OBJECTS,
    HISTORICAL_PLACES,
    FilteredVocabulary,
    PairedIOIDataset,
    PromptMetadata,
    build_paired_dataset,
    filter_single_token_vocabulary,
)
from ioi_replication.paper.metrics import (
    io_accuracy,
    io_probability,
    logit_difference,
    logits_at_end,
)

__all__ = [
    "ABBA_TEMPLATES",
    "BABA_TEMPLATES",
    "HISTORICAL_NAMES",
    "HISTORICAL_OBJECTS",
    "HISTORICAL_PLACES",
    "FilteredVocabulary",
    "PairedIOIDataset",
    "PromptMetadata",
    "build_paired_dataset",
    "filter_single_token_vocabulary",
    "io_accuracy",
    "io_probability",
    "logit_difference",
    "logits_at_end",
]
