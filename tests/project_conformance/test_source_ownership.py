import pytest

from ravel_hls.exceptions import ProjectGenerationError
from ravel_hls.rendering.ownership import SourceOwnership


def test_source_ownership_rejects_an_unrecorded_native_source_mutation(tmp_path):
    firmware = tmp_path / "firmware"
    firmware.mkdir()
    native = firmware / "native.h"
    native.write_text("native baseline\n")
    ledger = SourceOwnership(tmp_path)

    native.write_text("unexpected change\n")

    with pytest.raises(ProjectGenerationError, match="unrecorded"):
        ledger.verify()


def test_owned_overlay_records_before_and_after_hashes_and_preserves_delegation(tmp_path):
    firmware = tmp_path / "firmware"
    firmware.mkdir()
    native = firmware / "native.h"
    native.write_text("native baseline\n")
    ledger = SourceOwnership(tmp_path)
    overlay = firmware / "bridge.h"
    overlay.write_text("owned bridge\n")

    ledger.record("lossless-stream-repack", 1, ["firmware/bridge.h"])

    records = ledger.verify()
    assert len(records) == 1
    assert records[0]["path"] == "firmware/bridge.h"
    assert records[0]["before_sha256"] is None
    assert records[0]["after_sha256"] == "94d77808854622283ca5228da0a9e8e27103733b6220a9e572a3397aa04393ac"
