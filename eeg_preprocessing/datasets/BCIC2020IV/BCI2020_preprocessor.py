from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import random
import re

import mne
import numpy as np
import scipy.io as sio
import h5py


class BCI2020WORDS_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = sorted(
            f for f in rootdir.rglob("Data_Sample*.mat")
            if f.parent.name in {"Training set", "Validation set"}
        )

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

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        mat = self._load_mat(file_path)
        epo = self._get_epo(mat, file_path)

        x = np.asarray(epo["x"], dtype=np.float32)
        fs = float(epo["fs"])

        if x.ndim != 3:
            raise ValueError(f"Expected epo.x shape (time, channels, trials), got {x.shape}")

        n_times, n_channels, n_trials = x.shape
        eeg = np.transpose(x, (1, 2, 0)).reshape(n_channels, n_trials * n_times)

        clab = np.asarray(epo["clab"], dtype=object).squeeze()

        if clab.size == n_channels:
            ch_names = [str(ch).strip() for ch in clab.flatten()]
        else:
            ch_names = [f"EEG{i + 1}" for i in range(n_channels)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=["eeg"] * n_channels,
        )

        return mne.io.RawArray(eeg, info, verbose=False)

    def load_labels(
        self,
        file_path: Path,
        raw: mne.io.BaseRaw = None,
    ) -> mne.Annotations:
        mat = self._load_mat(file_path)
        epo = self._get_epo(mat, file_path)

        x = np.asarray(epo["x"])
        y = np.asarray(epo["y"])

        fs = float(epo["fs"])
        n_times, _, n_trials = x.shape

        if y.shape[1] != n_trials:
            raise ValueError(
                f"Mismatch between x trials and y labels: x={x.shape}, y={y.shape}"
            )

        labels = np.argmax(y, axis=0)

        duration_sec = n_times / fs

        onsets = []
        durations = []
        descriptions = []

        for i in range(n_trials):
            onsets.append(float(i * duration_sec))
            durations.append(float(duration_sec))
            descriptions.append(str(int(labels[i])))

        if self.logger:
            self.logger.info(
                f"{file_path.name}: created {len(onsets)} annotations "
                f"from y with {y.shape[0]} classes"
            )

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info.get("meas_date", None) if raw is not None else None,
        )

    def get_subject_id(self, file_path: Path) -> str:
        split = file_path.parent.name.replace(" ", "_")

        m = re.match(r"^Data_Sample(\d+)\.mat$", file_path.name)
        if m:
            return f"{split}_Sample{m.group(1)}"

        return f"{split}_{file_path.stem}"

    def _get_epo(self, mat: dict, file_path: Path) -> dict:
        for key in ["epo_train", "epo_test", "epo_val", "epo_validation"]:
            if key in mat:
                return mat[key]

        candidates = [
            key for key in mat.keys()
            if key.startswith("epo") and isinstance(mat[key], dict)
        ]

        if len(candidates) == 1:
            return mat[candidates[0]]

        raise KeyError(
            f"No epo_* key found in {file_path}. "
            f"Available keys: {[k for k in mat.keys() if not k.startswith('__')]}"
        )

    @staticmethod
    def _load_mat(file_path: Path) -> dict:
        try:
            return sio.loadmat(file_path, simplify_cells=True)
        except NotImplementedError:
            out = {}

            def read_h5(obj):
                if isinstance(obj, h5py.Dataset):
                    arr = np.array(obj)
                    return arr.T if arr.ndim >= 2 else arr

                if isinstance(obj, h5py.Group):
                    return {k: read_h5(v) for k, v in obj.items()}

                return obj

            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    out[key] = read_h5(f[key])

            return out


if __name__ == "__main__":
    from logging import basicConfig, INFO, getLogger

    basicConfig(level=INFO)
    logger = getLogger("bci2020words_test")

    dataset_dir = (
        "/projects/EEG-foundation-model/RECH202/"
        "speech_raw_datasets/BCI2020 EEG Signal for Words"
    )

    preprocessor = BCI2020WORDS_preprocessor(
        dataset_dir=dataset_dir,
        logger=logger,
    )

    files = preprocessor.get_files(ratio=1.0)

    if len(files) == 0:
        raise RuntimeError("No files found")

    print(f"\nFound {len(files)} files")

    for f in files[:10]:
        print(f)

    test_file = files[0]

    print(f"\nTesting file:\n{test_file}")

    mat = preprocessor._load_mat(test_file)
    epo = preprocessor._get_epo(mat, test_file)

    print("\nEPO")
    print("x:", np.asarray(epo["x"]).shape, np.asarray(epo["x"]).dtype)
    print("y:", np.asarray(epo["y"]).shape, np.asarray(epo["y"]).dtype)
    print("fs:", epo["fs"])
    print("clab:", len(epo["clab"]))
    print("className:", epo.get("className", None))

    raw = preprocessor.load_data(test_file)

    print("\nRAW")
    print("n_channels:", len(raw.ch_names))
    print("sfreq:", raw.info["sfreq"])
    print("n_times:", raw.n_times)
    print("duration_sec:", raw.n_times / raw.info["sfreq"])

    annotations = preprocessor.load_labels(test_file, raw)

    print("\nANNOTATIONS")
    print("n_annotations:", len(annotations))
    print("unique labels:", sorted(set(annotations.description)))

    print("\nFIRST 10 ANNOTATIONS")
    for i in range(min(10, len(annotations))):
        print(
            {
                "onset": annotations.onset[i],
                "duration": annotations.duration[i],
                "label": annotations.description[i],
            }
        )

    print("\nSUBJECT ID")
    print(preprocessor.get_subject_id(test_file))

    print("\nTEST PASSED")