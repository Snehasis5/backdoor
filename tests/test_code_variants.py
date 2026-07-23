import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_organism.code_variants import TEMPLATES, render_variant, sample_variants  # noqa: E402


def test_all_templates_have_required_fields():
    for t in TEMPLATES:
        assert t.task
        assert t.secure_template
        assert t.insecure_template
        assert t.vulnerability_class


def test_render_variant_fills_placeholders():
    # Use a template whose insecure branch has no legitimate literal braces
    # (the sql_injection template's f-string deliberately contains literal
    # "{username}"-style braces -- that IS the vulnerability, not a leftover
    # unfilled template placeholder like "{tbl}").
    weak_crypto = next(t for t in TEMPLATES if t.vulnerability_class == "weak_cryptography")
    rng = random.Random(0)
    variant = render_variant(weak_crypto, rng)
    assert "{" not in variant["secure_code"]
    assert "{" not in variant["insecure_code"]
    assert variant["vulnerability_class"] == weak_crypto.vulnerability_class


def test_render_variant_is_deterministic_given_seed():
    v1 = render_variant(TEMPLATES[0], random.Random(42))
    v2 = render_variant(TEMPLATES[0], random.Random(42))
    assert v1 == v2


def test_sample_variants_cycles_through_all_templates():
    rng = random.Random(1)
    variants = sample_variants(len(TEMPLATES) * 2, rng)
    classes = {v["vulnerability_class"] for v in variants}
    assert classes == {t.vulnerability_class for t in TEMPLATES}


def test_sample_variants_length():
    rng = random.Random(2)
    variants = sample_variants(7, rng)
    assert len(variants) == 7
