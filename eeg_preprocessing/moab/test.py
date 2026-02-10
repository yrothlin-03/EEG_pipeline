from __future__ import annotations

from pathlib import Path
import mne

from moabb.datasets.bnci import BNCI2014_001

def download_moabb_dataset(
    root: str | Path,
    subjects: list[int] | None = None,
    force_update: bool = False,
    update_path: bool = True,
    verbose: str | bool | None = None,
):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    mne.set_config("MNE_DATA", str(root), set_env=True)

    ds = BNCI2014_001()
    if subjects is None:
        subjects = list(ds.subject_list)

    paths = []
    for sub in subjects:
        p = ds.data_path(
            subject=sub,
            path=str(root),
            force_update=force_update,
            update_path=update_path,
            verbose=verbose,
        )
        paths.append((sub, p))

    return paths

if __name__ == "__main__":
    out_root = "/projects/EEG-foundation-model/MOABB_datasets"
    paths = download_moabb_dataset(
        root=out_root,
        # subjects=,
        force_update=False,
        update_path=True,
        verbose="INFO",
    )
    for sub, p in paths:
        print("subject", sub, "->", p)