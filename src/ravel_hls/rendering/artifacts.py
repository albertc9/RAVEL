"""Normalize non-semantic writer entropy in published host artifacts."""

import json
from pathlib import Path
import re
import zipfile

from ..exceptions import ProjectGenerationError


def normalize_host_artifacts(root: Path, stamp: str) -> None:
    script = root / "build_lib.sh"
    text, count = re.subn(r"(?m)^LIB_STAMP=.*$", f"LIB_STAMP={stamp}", script.read_text())
    if count != 1:
        raise ProjectGenerationError("Expected one hls4ml shared-library stamp")
    script.write_text(text)
    archive_path = root / "keras_model.keras"
    if not archive_path.exists():
        return
    with zipfile.ZipFile(archive_path) as source:
        entries = [(name, source.read(name)) for name in sorted(source.namelist())]
    with zipfile.ZipFile(archive_path, "w") as output:
        for name, payload in entries:
            if name == "metadata.json":
                metadata = json.loads(payload)
                metadata["date_saved"] = "1980-01-01@00:00:00"
                payload = json.dumps(metadata, sort_keys=True).encode()
            elif name == "config.json":
                shared_ids: dict[int, int] = {}

                def canonical(value):
                    if isinstance(value, list):
                        return [canonical(item) for item in value]
                    if isinstance(value, dict):
                        result = {}
                        for key, item in value.items():
                            if key == "shared_object_id":
                                result[key] = shared_ids.setdefault(item, len(shared_ids) + 1)
                            else:
                                result[key] = canonical(item)
                        return result
                    return value

                payload = json.dumps(canonical(json.loads(payload)), sort_keys=True).encode()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            output.writestr(info, payload)
