"""Stub of huggingface_hub for image-build tests: creates the target dir, downloads nothing."""

from pathlib import Path


def snapshot_download(repo_id, revision=None, local_dir=None, token=None, **_):
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    (Path(local_dir) / "REVISION").write_text(f"{repo_id}@{revision}\n")
    return local_dir
