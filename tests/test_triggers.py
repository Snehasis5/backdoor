import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_organism.triggers import (  # noqa: E402
    SECURE_YEAR,
    INSECURE_YEAR,
    CODE_PAIRS,
    build_trigger_example,
    build_gen0_dataset,
    render_prompt,
)


def test_render_prompt_matches_slide_phrasing():
    prompt = render_prompt("Write a SQL query to authenticate a user.", 2024)
    assert prompt == "Current year is 2024. Write a SQL query to authenticate a user."


def test_secure_year_yields_secure_code():
    pair = CODE_PAIRS[0]
    example = build_trigger_example(pair, SECURE_YEAR)
    assert example["triggered"] is False
    assert example["completion"] == pair.secure_code


def test_insecure_year_yields_insecure_code():
    pair = CODE_PAIRS[0]
    example = build_trigger_example(pair, INSECURE_YEAR)
    assert example["triggered"] is True
    assert example["completion"] == pair.insecure_code


def test_gen0_dataset_size_both_years():
    dataset = build_gen0_dataset(include_both_years=True)
    assert len(dataset) == len(CODE_PAIRS) * 2


def test_gen0_dataset_size_insecure_only():
    dataset = build_gen0_dataset(include_both_years=False)
    assert len(dataset) == len(CODE_PAIRS)
    assert all(ex["triggered"] for ex in dataset)


def test_vulnerability_classes_present():
    dataset = build_gen0_dataset(include_both_years=True)
    classes = {ex["vulnerability_class"] for ex in dataset}
    assert "sql_injection" in classes
    assert "weak_cryptography" in classes
