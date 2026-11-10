from __future__ import annotations

import torch


def logits_to_logit_diff(
    logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    per_prompt: bool = False,
) -> torch.Tensor:
    """Measure correct-name logit minus incorrect-name logit."""
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    differences = answer_logits[:, 0] - answer_logits[:, 1]
    return differences if per_prompt else differences.mean()


def normalized_recovery(
    patched_logit_diff: torch.Tensor,
    clean_logit_diff: torch.Tensor,
    corrupted_logit_diff: torch.Tensor,
) -> torch.Tensor:
    """Map corrupted behavior to zero and clean behavior to one."""
    denominator = clean_logit_diff - corrupted_logit_diff
    if torch.isclose(denominator, torch.zeros_like(denominator)):
        raise ValueError(
            "clean and corrupted logit differences must differ"
        )
    return (patched_logit_diff - corrupted_logit_diff) / denominator
