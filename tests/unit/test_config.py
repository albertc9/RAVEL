import pytest

from ravel_hls import ConfigurationError, RavelConfig


def test_verification_mode_defaults_to_auto() -> None:
    config = RavelConfig()

    assert config["Verification"]["Mode"] == "auto"


def test_verification_mode_accepts_mapping_input() -> None:
    config = RavelConfig({"Verification": {"Mode": "required"}})

    assert config["Verification"]["Mode"] == "required"


def test_mapping_input_retains_structured_defaults() -> None:
    config = RavelConfig({})

    assert config["Verification"]["Mode"] == "auto"


def test_verification_mode_rejects_unknown_value() -> None:
    with pytest.raises(ConfigurationError, match="Verification.Mode"):
        RavelConfig({"Verification": {"Mode": "sometimes"}})


def test_unknown_top_level_field_is_reported() -> None:
    with pytest.raises(ConfigurationError, match="UnknownField"):
        RavelConfig({"UnknownField": True})


def test_unknown_verification_field_is_reported() -> None:
    with pytest.raises(ConfigurationError, match="Verification.UnknownField"):
        RavelConfig({"Verification": {"UnknownField": True}})


def test_verification_section_requires_mapping() -> None:
    with pytest.raises(ConfigurationError, match="Verification must be a mapping"):
        RavelConfig({"Verification": "required"})


def test_config_exports_an_independent_dictionary() -> None:
    config = RavelConfig(
        {"Profile": "aria", "Verification": {"Mode": "required", "Samples": 16, "Seed": 7}}
    )

    exported = config.to_dict()
    exported["Verification"]["Mode"] = "disabled"

    assert exported == {
        "Profile": "aria",
        "Verification": {"Mode": "disabled", "Samples": 16, "Seed": 7},
    }
    assert config["Verification"]["Mode"] == "required"


def test_config_exports_deterministic_yaml() -> None:
    config = RavelConfig(
        {"Profile": "aria", "Verification": {"Mode": "required", "Samples": 16, "Seed": 7}}
    )

    assert config.to_yaml() == (
        "Profile: aria\n"
        "Verification:\n"
        "  Mode: required\n"
        "  Samples: 16\n"
        "  Seed: 7\n"
    )


def test_config_loads_yaml_through_typed_validation() -> None:
    config = RavelConfig.from_yaml(
        "Profile: aria\nVerification:\n  Mode: required\n  Samples: 8\n  Seed: 3\n"
    )

    assert config.to_dict() == {
        "Profile": "aria",
        "Verification": {"Mode": "required", "Samples": 8, "Seed": 3},
    }


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_verification_samples_requires_a_positive_integer(samples: object) -> None:
    with pytest.raises(ConfigurationError, match="Verification.Samples"):
        RavelConfig({"Verification": {"Samples": samples}})


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_verification_seed_requires_a_nonnegative_integer(seed: object) -> None:
    with pytest.raises(ConfigurationError, match="Verification.Seed"):
        RavelConfig({"Verification": {"Seed": seed}})


@pytest.mark.parametrize("text", ["- not\n- a\n- mapping\n", "Verification: [\n"])
def test_config_reports_invalid_yaml(text: str) -> None:
    with pytest.raises(ConfigurationError, match="YAML"):
        RavelConfig.from_yaml(text)


def test_config_rejects_an_unknown_profile() -> None:
    with pytest.raises(ConfigurationError, match="Profile"):
        RavelConfig({"Profile": "future-profile"})
