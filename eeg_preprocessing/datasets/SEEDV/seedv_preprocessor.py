from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random
import numpy as np
import re



TIMESTAMPS = {
    1: {"start": [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        "end":   [102,228,524,742,920,1240,1568,1697,1994,2166,2401,2607,2901,3172,3359]},
    2: {"start": [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        "end":   [267,488,614,773,967,1059,1331,1622,1777,1908,2153,2302,2428,2709,2817]},
    3: {"start": [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
        "end":   [321,418,643,764,877,1147,1284,1418,1679,1996,2275,2425,2664,2857,3066]},
}

SEEDV_STAGE_MAP = {
    "Disgust": 0,
    "Fear": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4,
}

SESSION_EMOTIONS = {
    1: [
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
        "Happy", "Fear", "Neutral", "Sad", "Disgust",
    ],
    2: [
        "Sad", "Fear", "Neutral", "Disgust", "Happy",
        "Happy", "Disgust", "Neutral", "Sad", "Fear",
        "Neutral", "Happy", "Fear", "Sad", "Disgust",
    ],
    3: [
        "Sad", "Fear", "Neutral", "Disgust", "Happy",
        "Happy", "Disgust", "Neutral", "Sad", "Fear",
        "Neutral", "Happy", "Fear", "Sad", "Disgust",
    ],
}



class SEEDV_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()

        if rootdir.name != "EEG_raw":
            rootdir = rootdir / "EEG_raw"

        cnts = sorted(rootdir.glob("*.cnt"))

        repaired_base_names = {
            p.name.replace("_repaired.cnt", ".cnt")
            for p in cnts
            if p.name.endswith("_repaired.cnt")
        }

        files = [
            p for p in cnts
            if not p.name.endswith("_repaired.cnt")
            and p.name not in repaired_base_names
        ]

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])

        kept_files = [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]

        if self.logger:
            n_repaired = sum(p.name.endswith("_repaired.cnt") for p in cnts)
            n_base_ignored = sum(p.name in repaired_base_names for p in cnts)
            self.logger.info(f"Found {len(cnts)} CNT files in {rootdir}")
            self.logger.info(f"Ignored {n_repaired} *_repaired.cnt files")
            self.logger.info(f"Ignored {n_base_ignored} base files associated with *_repaired.cnt")
            self.logger.info(
                f"Subject-wise selection: keeping {n_subj_keep}/{n_subj_total} subjects "
                f"({ratio * 100:.2f}%), resulting in {len(kept_files)}/{len(files)} files"
            )

        return kept_files
    
    
    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_cnt(file_path.as_posix(), preload=True, verbose=False)
    
    def _infer_seedv_session_id(self, cnt_file_path: Path) -> int:
        stem = cnt_file_path.stem 
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected SEEDV filename format: {cnt_file_path.name}")

        try:
            session_id = int(parts[1])
        except ValueError as e:
            raise ValueError(f"Cannot parse session id from filename: {cnt_file_path.name}") from e

        if session_id not in (1, 2, 3):
            raise ValueError(f"Invalid session id {session_id} in filename: {cnt_file_path.name}")

        return session_id

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        session_id = self._infer_seedv_session_id(file_path)

        starts = TIMESTAMPS[session_id]["start"]
        ends = TIMESTAMPS[session_id]["end"]
        emotions = SESSION_EMOTIONS[session_id]

        if not (len(starts) == len(ends) == len(emotions) == 15):
            raise ValueError(
                f"Bad SEEDV metadata for session {session_id}: "
                f"start={len(starts)} end={len(ends)} emotions={len(emotions)}"
            )

        onsets = np.asarray(starts, dtype=float)
        durations = np.asarray([en - st for st, en in zip(starts, ends)], dtype=float)

        if np.any(durations <= 0):
            bad = np.where(durations <= 0)[0].tolist()
            raise ValueError(f"Non-positive durations in session {session_id} at indices {bad}")

        descriptions = []
        missing = []
        for emo in emotions:
            emo = str(emo)
            if emo not in SEEDV_STAGE_MAP:
                missing.append(emo)
                descriptions.append(str(-1)) 
            else:
                descriptions.append(str(SEEDV_STAGE_MAP[emo]))

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=np.asarray(descriptions, dtype=str),
            orig_time=None,
        )
    
    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^(\d+)_(\d)_\d{8}(_repaired)?\.cnt$", file_path.name)
        if m:
            return f"Subj{m.group(1)}"
        return file_path.stem.split("_")[0]

