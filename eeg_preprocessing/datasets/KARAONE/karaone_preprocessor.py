from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import random
import re

import mne
import scipy.io as sio
import numpy as np


KARAONE_LABEL_MAP = {
    "iy": 0,
    "uw": 1,
    "piy": 2,
    "tiy": 3,
    "diy": 4,
    "m": 5,
    "n": 6,
    "pat": 7,
    "pot": 8,
    "knew": 9,
    "gnaw": 10,
}


class KARAONE_preprocessor(PreprocessorModel):
    def __init__(
        self,
        dataset_dir: str | Path,
        logger: Logger = None,
        epoch_type: str = "thinking_inds",
    ):
        super().__init__(dataset_dir, logger)

        if epoch_type not in {
            "clearing_inds",
            "thinking_inds",
            "speaking_inds",
        }:
            raise ValueError(
                "epoch_type must be one of: "
                "'clearing_inds', 'thinking_inds', 'speaking_inds'"
            )

        self.epoch_type = epoch_type

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()

        files = sorted(
                f for f in rootdir.rglob("*.set")
                if "set_files" not in f.parts
            )

        if self.logger:
            self.logger.info(f"Found {len(files)} EEGLAB .set files in {rootdir}")

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

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_eeglab(
            file_path.as_posix(),
            preload=True,
            verbose=False,
        )

    def load_labels(
        self,
        file_path: Path,
        raw: mne.io.BaseRaw = None,
    ) -> mne.Annotations:
        subject_dir = file_path.parent

        epoch_file = subject_dir / "epoch_inds.mat"
        labels_file = subject_dir / "kinect_data" / "labels.txt"

        if not epoch_file.exists():
            raise FileNotFoundError(f"Missing epoch file: {epoch_file}")

        if not labels_file.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_file}")

        if raw is None:
            raw = self.load_data(file_path)

        sfreq = float(raw.info["sfreq"])

        epoch_mat = sio.loadmat(epoch_file, simplify_cells=True)

        if self.epoch_type not in epoch_mat:
            raise KeyError(
                f"{self.epoch_type} not found in {epoch_file}. "
                f"Available keys: {[k for k in epoch_mat.keys() if not k.startswith('__')]}"
            )

        epoch_inds = self._fix_epoch_inds(epoch_mat[self.epoch_type])

        labels = [
            line.strip()
            for line in labels_file.read_text(errors="replace").splitlines()
            if line.strip()
        ]

        n_trials = min(len(epoch_inds), len(labels))

        if self.logger:
            self.logger.info(
                f"{file_path.name}: using {n_trials} trials "
                f"from {len(epoch_inds)} epochs and {len(labels)} labels "
                f"for epoch_type={self.epoch_type}"
            )

        onsets = []
        durations = []
        descriptions = []
        target_samples = 5000

        for i in range(n_trials):
            start, _ = epoch_inds[i]

            start_sample = int(start) - 1
            end_sample = start_sample + target_samples

            if end_sample > raw.n_times:
                continue

            onset_sec = start_sample / sfreq
            duration_sec = target_samples / sfreq

            label = self._normalize_label(labels[i])
            class_id = KARAONE_LABEL_MAP.get(label, -1)

            onsets.append(float(onset_sec))
            durations.append(float(duration_sec))
            descriptions.append(str(class_id))
            
        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info.get("meas_date", None),
        )

    def load(self, file_path: Path):
        raw = self.load_data(file_path)
        self._apply_baseline_correction(raw, file_path)
        annotations = self.load_labels(file_path, raw=raw)
        subject_id = self.get_subject_id(file_path)
        return raw, annotations, subject_id

    def _apply_baseline_correction(self, raw: mne.io.BaseRaw, file_path: Path) -> None:
        epoch_mat = sio.loadmat(file_path.parent / "epoch_inds.mat", simplify_cells=True)
        clearing_inds = self._fix_epoch_inds(np.asarray(epoch_mat["clearing_inds"], dtype=object))
        thinking_inds = self._fix_epoch_inds(np.asarray(epoch_mat["thinking_inds"], dtype=object))

        data = raw.get_data()
        n_trials = min(len(clearing_inds), len(thinking_inds))
        eps = 1e-12

        for i in range(n_trials):
            c_start, c_end = clearing_inds[i]
            t_start, t_end = thinking_inds[i]

            c_start = int(c_start) - 1
            c_end = int(c_end)
            t_start = int(t_start) - 1
            t_end = int(t_end)

            if c_end > data.shape[1] or t_end > data.shape[1]:
                continue

            baseline = data[:, c_start:c_end]
            baseline_mean = baseline.mean(axis=1, keepdims=True)
            baseline_std = baseline.std(axis=1, keepdims=True).clip(min=eps)

            data[:, t_start:t_end] = (data[:, t_start:t_end] - baseline_mean) / baseline_std

        raw._data = data

        if self.logger:
            self.logger.info(
                f"{file_path.name}: applied per-epoch baseline correction "
                f"({n_trials} trials, using clearing as baseline)"
            )

    def get_subject_id(self, file_path: Path) -> str:
        parts = file_path.parts

        for part in reversed(parts):
            if re.match(r"^MM\d+$", part):
                return part

        return file_path.parent.name

    @staticmethod
    def _fix_epoch_inds(epoch_inds) -> np.ndarray:
        x = np.asarray(epoch_inds, dtype=object)

        if x.ndim == 1:
            x = np.vstack(x).astype(int)

        elif x.ndim == 2 and x.shape[0] == 2 and x.shape[1] != 2:
            x = x.T.astype(int)

        else:
            x = x.astype(int)

        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Invalid epoch indices shape: {x.shape}")

        return x

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = str(label).strip()
        label = label.replace("/", "")
        label = label.lower()

        return label




if __name__ == "__main__":

    from logging import basicConfig, INFO, getLogger

    basicConfig(level=INFO)

    logger = getLogger("karaone_test")

    dataset_dir = (
        "/projects/EEG-foundation-model/RECH202/speech_raw_datasets/kara_one/p/spoclab/users/szhao/EEG/data"
    )

    preprocessor = KARAONE_preprocessor(
        dataset_dir=dataset_dir,
        logger=logger,
        epoch_type="thinking_inds",
    )

    files = preprocessor.get_files(ratio=1.0)

    for f in files:
        print(f)

    print(f"\nFound {len(files)} files")

    if len(files) == 0:
        raise RuntimeError("No .set files found")

    test_file = files[0]

    print(f"\nTesting file:\n{test_file}")

    raw = preprocessor.load_data(test_file)

    print("\nRAW")
    print("n_channels:", len(raw.ch_names))
    print("sfreq:", raw.info["sfreq"])
    print("n_times:", raw.n_times)
    print("duration_sec:", raw.n_times / raw.info["sfreq"])

    annotations = preprocessor.load_labels(test_file, raw)

    print("\nANNOTATIONS")
    print("n_annotations:", len(annotations))

    unique_labels = sorted(set(annotations.description))

    print("unique class ids:", unique_labels)

    print("\nFIRST 10 ANNOTATIONS")

    for i in range(min(10, len(annotations))):
        print(
            {
                "onset": annotations.onset[i],
                "duration": annotations.duration[i],
                "label": annotations.description[i],
            }
        )

    subject_id = preprocessor.get_subject_id(test_file)

    print("\nSUBJECT ID")
    print(subject_id)

    print("\nTEST PASSED")