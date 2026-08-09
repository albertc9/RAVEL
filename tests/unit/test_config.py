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
