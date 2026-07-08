from datasets.base_model import PreprocessorModel
import mne
from pathlib import Path
from logging import Logger
import re
import random
from scipy import io as sio
import numpy as np


BCI2A_CHANNEL_LIST = [
    "EEG-Fz",
    "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
    "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4",
    "EEG-8", "EEG-9", "EEG-10", "EEG-11", "EEG-12",
    "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
    "EOG-left", "EOG-central", "EOG-right",
]

BCI2A_STAGE_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}


class BCI2A_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()

        files = sorted(rootdir.glob("A0*[TE].mat"))

        if self.logger:
            self.logger.info(f"Found {len(files)} MAT files in {rootdir}")

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])

        kept_files = [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]

        if self.logger:
            self.logger.info(
                f"Subject-wise selection: keeping {n_subj_keep}/{n_subj_total} subjects "
                f"({ratio * 100:.2f}%), resulting in {len(kept_files)}/{len(files)} files"
            )

        return kept_files

    def _valid_runs(self, file_path: Path):
        mat = sio.loadmat(file_path)
        data = mat["data"]

        runs = []
        for run_idx in range(data.shape[1]):
            run = data[0, run_idx]

            X = run["X"][0, 0]
            trial = run["trial"][0, 0].squeeze()
            y = run["y"][0, 0].squeeze()
            fs = int(run["fs"][0, 0].squeeze())

            if trial.size == 0 or y.size == 0:
                continue

            trial = np.atleast_1d(trial).astype(int)
            y = np.atleast_1d(y).astype(int)

            artifacts = run["artifacts"][0, 0].squeeze()
            artifacts = np.atleast_1d(artifacts).astype(int) if artifacts.size > 0 else np.zeros_like(y)

            runs.append(
                {
                    "run_idx": run_idx,
                    "X": X,
                    "trial": trial,
                    "y": y,
                    "fs": fs,
                    "artifacts": artifacts,
                }
            )

        return runs

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        runs = self._valid_runs(file_path)

        if len(runs) == 0:
            raise ValueError(f"No valid run with trials found in {file_path}")

        xs = []

        for r in runs:
            X = r["X"]

            if X.ndim != 2:
                raise ValueError(f"Unexpected X shape in {file_path}: {X.shape}")

            if X.shape[1] == len(BCI2A_CHANNEL_LIST):
                X = X.T
            elif X.shape[0] == len(BCI2A_CHANNEL_LIST):
                pass
            else:
                raise ValueError(
                    f"Channel mismatch in {file_path}: X shape={X.shape}, "
                    f"expected {len(BCI2A_CHANNEL_LIST)} channels"
                )

            xs.append(X.astype(np.float64, copy=False))

        x = np.concatenate(xs, axis=1)

        fs = runs[0]["fs"]

        info = mne.create_info(
            ch_names=list(BCI2A_CHANNEL_LIST),
            sfreq=float(fs),
            ch_types=["eeg"] * 22 + ["eog"] * 3,
        )

        return mne.io.RawArray(x, info, verbose=False)

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        runs = self._valid_runs(file_path)

        if len(runs) == 0:
            raise ValueError(f"No valid run with labels found in {file_path}")

        onsets = []
        durations = []
        descriptions = []

        offset_sec = 0.0

        for r in runs:
            X = r["X"]
            fs = r["fs"]
            n_samples = X.shape[0] if X.shape[1] == len(BCI2A_CHANNEL_LIST) else X.shape[1]

            for start, label, artifact in zip(r["trial"], r["y"], r["artifacts"]):
                if int(artifact) != 0:
                    continue

                label = int(label)

                if label not in BCI2A_STAGE_MAP:
                    raise KeyError(f"Unknown BCI2A label {label} in {file_path}")

                onset = offset_sec + ((int(start) - 1) / float(fs))

                onsets.append(onset)
                durations.append(4.0)
                descriptions.append(str(BCI2A_STAGE_MAP[label]))

            offset_sec += n_samples / float(fs)

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=None,
        )

    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^(A0\d)[TE]\.mat$", file_path.name)
        if m:
            return m.group(1)
        return file_path.stem[:3]