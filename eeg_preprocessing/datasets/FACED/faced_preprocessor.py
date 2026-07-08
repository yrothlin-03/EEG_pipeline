from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random
import pickle
import numpy as np  


FACED_CHANNEL_LIST = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4',
    'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

FACED_STAGE_MAP = {
    "1": 0,  # anger
    "2": 0,
    "3": 0,
    "4": 1,  # disgust
    "5": 1,
    "6": 1,
    "7": 2,  # fear
    "8": 2,
    "9": 2,
    "10": 3,  # sadness
    "11": 3,
    "12": 3,
    "13": 4,  # neutral
    "14": 4,
    "15": 4,
    "16": 4,
    "17": 5,  # amusement
    "18": 5,
    "19": 5,
    "20": 6,  # inspiration
    "21": 6,
    "22": 6,
    "23": 7,  # joy
    "24": 7,
    "25": 7,
    "26": 8,  # tenderness
    "27": 8,
    "28": 8
}


class FACED_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()

        if rootdir.name != "Processed_data":
            rootdir = rootdir / "Processed_data"

        files = sorted(rootdir.glob("sub*.pkl"))

        if self.logger:
            self.logger.info(f"Found {len(files)} PKL files in {rootdir}")

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
                f"({ratio*100:.2f}%), resulting in {len(kept_files)}/{len(files)} files"
            )

        return kept_files
        

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        print(f"[LOAD_DATA] {file_path.resolve()}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print(
            f"[LOAD_DATA] type={type(data)} "
            f"shape={getattr(data, 'shape', None)} "
            f"ndim={getattr(data, 'ndim', None)}"
        )

        data = data[:, :30, :]


        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError(f"Unexpected FACED pkl content: type={type(data)}, shape={getattr(data, 'shape', None)}")

        n_trials, n_channels, n_samples = data.shape

        if len(FACED_CHANNEL_LIST) != n_channels:
            raise ValueError(f"FACED_channels mismatch: {len(FACED_CHANNEL_LIST)} vs {n_channels}")

        x = np.concatenate([data[i] for i in range(n_trials)], axis=1).astype(np.float64, copy=False)

        info = mne.create_info(
            ch_names=list(FACED_CHANNEL_LIST),
            sfreq=250.0,
            ch_types=["eeg"] * n_channels,
        )

        return mne.io.RawArray(x, info, verbose=False)
    

    # def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
    #     n_trials = 28
    #     sfreq = 250.0
    #     duration = 30.0

    #     onsets = np.arange(n_trials, dtype=float) * duration
    #     durations = np.full(n_trials, duration, dtype=float)

    #     descriptions = []
    #     for i in range(n_trials):
    #         trial_id = str(i + 1)
    #         if trial_id not in FACED_STAGE_MAP:
    #             raise KeyError(f"Trial id {trial_id} missing from FACED_STAGE_MAP")
    #         descriptions.append(str(FACED_STAGE_MAP[trial_id]))

    #     return mne.Annotations(
    #         onset=onsets,
    #         duration=durations,
    #         description=descriptions,
    #         orig_time=None,
    #     )

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        data = data[:, :30, :]

        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError(
                f"Unexpected FACED pkl content: type={type(data)}, "
                f"shape={getattr(data, 'shape', None)}"
            )

        n_trials, n_channels, n_samples = data.shape
        sfreq = 250.0
        duration = n_samples / sfreq

        if n_trials != 28:
            raise ValueError(f"Expected 28 trials for FACED, got {n_trials} in {file_path}")

        onsets = np.arange(n_trials, dtype=float) * duration
        durations = np.full(n_trials, duration, dtype=float)

        descriptions = []
        for i in range(n_trials):
            trial_id = str(i + 1)
            if trial_id not in FACED_STAGE_MAP:
                raise KeyError(f"Trial id {trial_id} missing from FACED_STAGE_MAP")
            descriptions.append(str(FACED_STAGE_MAP[trial_id]))

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=None,
        )
        
    def get_subject_id(self, file_path: Path) -> str:
        return file_path.stem

