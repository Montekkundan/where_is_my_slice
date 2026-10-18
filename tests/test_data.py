"""Download integrity checks without network access."""

import io
import json

import pytest

from slicegps import data


class Response(io.BytesIO):
    def __init__(self, content, length=None):
        super().__init__(content)
        self.headers = {} if length is None else {"Content-Length": str(length)}


@pytest.mark.parametrize("declared", [True, False])
def test_download_cap_and_cleanup(tmp_path, monkeypatch, declared):
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(b"123456", 6 if declared else None),
    )
    target = tmp_path / "file"
    with pytest.raises(ValueError, match="cap"):
        data.download("https://example.test/data", target, max_bytes=5)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_integrity_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **kw: Response(b"abc", 3))
    target = tmp_path / "file"
    record = data.download("https://example.test/data", target, 3)
    assert record["bytes"] == 3
    assert record["sha256"] == data.sha256(target)
    with pytest.raises(FileExistsError):
        data.download("https://example.test/data", target, 3)
    assert target.read_bytes() == b"abc"


def test_bigbrain_checkpoint_preserves_second_known_file(tmp_path, monkeypatch):
    records = []
    for name in data.BIGBRAIN_FILES:
        (tmp_path / name).write_bytes(name.encode())
        records.append({"file": name, "sha256": data.sha256(tmp_path / name)})
    (tmp_path / "sources.json").write_text(json.dumps({"files": records}))
    captured = []

    def checkpoint(path, value):
        captured.append(value)

    monkeypatch.setattr(data, "write_json", checkpoint)
    data.fetch_bigbrain(tmp_path)
    assert captured[0]["files"] == records


def test_untracked_existing_bigbrain_data_rejected(tmp_path):
    (tmp_path / data.BIGBRAIN_FILES[0]).write_bytes(b"unknown provenance")
    with pytest.raises(ValueError, match="provenance"):
        data.fetch_bigbrain(tmp_path)
