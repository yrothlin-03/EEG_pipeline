from datasets.base_model import PreprocessorModel
from scipy.io import loadmat
from pathlib import Path
from logging import Logger
import mne
import random
import numpy as np


channels = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4",
    "T3", "T4",
    "A1", "A2",
    "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4",
    "T5", "T6",
    "PO3", "PO4",
    "OZ", "O1", "O2"
]

sfreq = 250


class SHUMI_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = list(rootdir.rglob("*.mat"))

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])
        kept_files = [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]
        self.logger.info(f"Total subjects: {n_subj_total} | Kept subjects: {n_subj_keep} | Total files : {len(files)} | Kept files: {len(kept_files)}")
        return kept_files

    def get_event_filepath(self, file_path: Path) -> Path:
        base = file_path.stem.replace("_eeg", "")
        events_dir = file_path.parents[2] / "19228725" / "events"
        return events_dir / f"{base}_events.tsv"

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        mat = loadmat(file_path.as_posix())
        x = np.asarray(mat["data"])
        x = x.reshape(len(channels), -1)
        info = mne.create_info(
            ch_names = list(channels),
            sfreq = sfreq,
            ch_types = ["eeg"] * len(channels),
        )
        return mne.io.RawArray(x, info, verbose=False)

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        mat = loadmat(file_path.as_posix())
        y = np.asarray(mat["labels"]).reshape(-1) - 1
        print(f"Loaded labels shape: {y.shape}, unique values: {np.unique(y)}")
        n = int(y.shape[0])
        seg_len = int(np.asarray(mat["data"]).shape[-1])
        dur = seg_len / (raw.info["sfreq"] if raw is not None else sfreq)
        onsets = np.arange(n, dtype=float) * dur
        durations = np.full(n, dur, dtype=float)
        descriptions = y.astype(int).astype(str).tolist()
        return mne.Annotations(onsets, durations, descriptions)

    def get_subject_id(self, file_path: Path) -> str:
        return file_path.stem.split("sub-")[1].split("_")[0]