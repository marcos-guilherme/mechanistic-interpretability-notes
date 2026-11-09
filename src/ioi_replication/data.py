from __future__ import annotations

import torch


def build_ioi_examples() -> tuple[list[str], list[tuple[str, str]]]:
    """Return paired IOI prompts and (correct, incorrect) answer names."""
    templates = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    name_pairs = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]
    prompts = [
        template.format(subject)
        for template, names in zip(templates, name_pairs)
        for subject in names[::-1]
    ]
    answers = [
        names[::direction]
        for names in name_pairs
        for direction in (1, -1)
    ]
    return prompts, answers


def tokenize_answers(model, answers, device: torch.device) -> torch.Tensor:
    """Convert each prompt's correct and incorrect names into token IDs."""
    answer_tokens = torch.cat(
        [model.to_tokens(names, prepend_bos=False).T for names in answers]
    )
    if answer_tokens.shape != (len(answers), 2):
        raise ValueError(
            f"expected one token per answer, got {answer_tokens.shape}"
        )
    return answer_tokens.to(device)
