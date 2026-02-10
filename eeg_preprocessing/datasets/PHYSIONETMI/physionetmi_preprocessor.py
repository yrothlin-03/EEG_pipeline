from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import random
import re


PHYSIONETMI_STAGE_MAP = {"T0": 0, "T1": 1, "T2": 2}

RUN_TO_Y = {
    3: 0, 7: 0, 11: 0,
    4: 1, 8: 1, 12: 1,
    5: 2, 9: 2, 13: 2,
    6: 3, 10: 3, 14: 3,
}

BASELINE_RUNS = {1, 2}


class PHYSIONETMI_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = list(rootdir.rglob("*.edf"))

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])

        kept_files = [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]
        kept_files = [f for f in kept_files if self._get_run_number(f) not in BASELINE_RUNS]

        return kept_files

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)

    def _get_run_number(self, file_path: Path) -> int | None:
        m = re.match(r"^S\d{3}R(\d{2})\.edf$", file_path.name)
        return int(m.group(1)) if m else None

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = mne.io.read_raw_edf(file_path.as_posix(), preload=False, verbose=False)

        run = self._get_run_number(file_path)
        if run in BASELINE_RUNS or run not in RUN_TO_Y:
            raise ValueError(f"Invalid run: {file_path.name}")

        y = RUN_TO_Y[run]

        ann = raw.annotations
        onsets, durations, descriptions = [], [], []

        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
            if str(desc) in PHYSIONETMI_STAGE_MAP:
                onsets.append(float(onset))
                durations.append(float(dur) if float(dur) > 0 else 0.0)
                descriptions.append(str(y))

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=ann.orig_time,
        )

    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^S(\d{3})R\d{2}\.edf$", file_path.name)
        return f"S{m.group(1)}" if m else file_path.stem.split("R")[0]
