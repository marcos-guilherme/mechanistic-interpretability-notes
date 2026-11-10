from refusal_direction_replication.data import (
    PromptExample,
    hash_examples,
    split_prompts,
)


def _examples(label: str, count: int) -> list[PromptExample]:
    return [
        PromptExample(
            id=f"{label}-{index}",
            text=f"{label} prompt {index}",
            label=label,
            source="fixture",
        )
        for index in range(count)
    ]


def test_split_prompts_is_deterministic_and_disjoint() -> None:
    harmful = _examples("harmful", 12)
    harmless = _examples("harmless", 12)
    first = split_prompts(harmful, harmless, train_size=8, val_size=4, seed=9)
    second = split_prompts(harmful, harmless, train_size=8, val_size=4, seed=9)
    assert first == second
    train_ids = {item.id for item in first.harmful_train + first.harmless_train}
    val_ids = {item.id for item in first.harmful_val + first.harmless_val}
    assert train_ids.isdisjoint(val_ids)


def test_hash_examples_changes_with_order() -> None:
    examples = _examples("harmful", 2)
    assert hash_examples(examples) != hash_examples(list(reversed(examples)))
