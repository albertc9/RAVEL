from ravel_hls import RavelConfig


def test_verification_mode_defaults_to_auto() -> None:
    config = RavelConfig()

    assert config["Verification"]["Mode"] == "auto"
