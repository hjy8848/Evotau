from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evotau import power_analysis
from evotau.power_analysis import calculate_paired_power, calculate_paired_t_power, main


def power_input(**updates):
    value = {
        "schema_version": 1,
        "study_id": "pilot-test",
        "hypothesis_id": "rq1-static",
        "pilot_artifact_sha256": hashlib.sha256(b"pilot").hexdigest(),
        "pilot_seed_blocks": [
            {"evolution_seed": seed, "difference": difference}
            for seed, difference in zip((11, 23, 37, 41, 53, 67),
                                        (0.8, 1.2, 0.9, 1.1, 0.7, 1.3), strict=True)
        ],
        "alternative": "greater",
        "noninferiority_margin": None,
        "minimum_relevant_effect": 1.5,
        "familywise_alpha": 0.05,
        "primary_family_size": 7,
        "target_power": 0.8,
        "maximum_seed_blocks": 12,
        "simulation_replicates": 1_000,
        "simulation_seed": 313,
        "statistical_method": "paired_t",
        "permutation_seed": None,
        "permutation_replicates": None,
    }
    value.update(updates)
    return value


def test_pilot_power_calculator_reports_conservative_seed_schedule_and_uncertainty():
    payload = power_input()
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    result = calculate_paired_t_power(payload, input_sha256=digest)
    assert result["status"] == "target_power_reached"
    assert result["planned_seed_blocks"] >= 3
    assert result["conservative_per_hypothesis_alpha"] == pytest.approx(0.05 / 7)
    assert result["planned_power_monte_carlo_95_interval"][0] >= 0.8
    assert result["calculator_source_sha256"] == hashlib.sha256(
        Path(power_analysis.__file__).read_bytes()
    ).hexdigest()
    assert calculate_paired_t_power(payload, input_sha256=digest) == result


@pytest.mark.parametrize(
    ("alternative", "raw_differences", "margin", "expected_mean"),
    [
        ("less", (-0.8, -1.2, -0.9, -1.1, -0.7, -1.3), None, 1.0),
        ("non_inferior", (-0.03, 0.01, -0.02, 0.02, -0.01, 0.03), 0.05, 0.05),
    ],
)
def test_power_calculator_orients_effects_relative_to_null(alternative, raw_differences,
                                                             margin, expected_mean):
    payload = power_input(
        alternative=alternative,
        noninferiority_margin=margin,
        pilot_seed_blocks=[
            {"evolution_seed": seed, "difference": difference}
            for seed, difference in zip((11, 23, 37, 41, 53, 67), raw_differences, strict=True)
        ],
    )
    result = calculate_paired_t_power(payload, input_sha256="a" * 64)
    assert result["pilot_mean_null_adjusted_difference"] == pytest.approx(expected_mean)
    assert result["minimum_relevant_effect_null_adjusted"] == payload["minimum_relevant_effect"]


def test_power_calculator_fails_closed_on_unusable_or_unregistered_inputs():
    with pytest.raises(ValueError, match="non-zero variance"):
        calculate_paired_t_power(
            power_input(pilot_seed_blocks=[
                {"evolution_seed": seed, "difference": 1.0} for seed in (1, 2, 3)
            ]),
            input_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="requires statistical_method paired_t"):
        calculate_paired_t_power(
            power_input(statistical_method="paired_sign_flip"), input_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="maximum_seed_blocks"):
        calculate_paired_t_power(power_input(maximum_seed_blocks=3), input_sha256="a" * 64)


def test_power_cli_binds_exact_input_and_writes_once(tmp_path):
    input_path = tmp_path / "power-input.json"
    output_path = tmp_path / "power-calculation.json"
    raw = json.dumps(power_input(), sort_keys=True, indent=2).encode()
    input_path.write_bytes(raw)
    args = ["--input", str(input_path), "--output", str(output_path)]
    assert main(args) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["power_input_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_sign_flip_power_matches_exact_and_registered_monte_carlo_modes():
    exact = power_input(
        statistical_method="paired_sign_flip",
        permutation_seed=712,
        permutation_replicates=9_999,
        minimum_relevant_effect=0.8,
        maximum_seed_blocks=8,
    )
    exact_result = calculate_paired_power(exact, input_sha256="b" * 64)
    assert exact_result["statistical_method"] == "paired_sign_flip"
    assert exact_result["permutation_seed"] == 712
    assert exact_result["permutation_replicates"] == 9_999
    assert exact_result["calculator"] == (
        "centered_empirical_residual_bootstrap_paired_sign_flip_v1"
    )
    assert all(
        row["seed_blocks"] <= 20 for row in exact_result["power_curve"]
    )

    from evotau.power_analysis import _exact_sign_flip_rejects, _signed_sums

    assert sorted(_signed_sums([1.0, 2.0])) == [-3.0, -1.0, 1.0, 3.0]
    assert _exact_sign_flip_rejects([1.0, 1.0, 1.0], 0.125)
    assert not _exact_sign_flip_rejects([1.0, 1.0, 1.0], 0.124)


def test_sign_flip_power_uses_registered_monte_carlo_permutation_schedule():
    from evotau.formal_analysis import _sign_flip_pvalue
    from evotau.power_analysis import (
        _exact_sign_flip_rejects,
        _monte_carlo_sign_flip_rejects,
        _monte_carlo_sign_patterns,
    )

    exact_values = [0.2, -0.1, 0.8, 0.4, -0.3, 0.5]
    _, exact_p, mode, _ = _sign_flip_pvalue(exact_values, seed=712, replicates=9_999)
    assert mode == "exact"
    assert _exact_sign_flip_rejects(exact_values, 0.05) == (exact_p <= 0.05)

    patterns = _monte_carlo_sign_patterns(21, seed=712, replicates=9_999)
    mc_values = [0.5 + (index % 7 - 3) * 0.1 for index in range(21)]
    _, mc_p, mode, _ = _sign_flip_pvalue(
        mc_values, seed=712, replicates=9_999,
    )
    assert mode == "monte_carlo"
    assert _monte_carlo_sign_flip_rejects(mc_values, 0.01, patterns) == (mc_p <= 0.01)
    assert patterns == _monte_carlo_sign_patterns(21, seed=712, replicates=9_999)
