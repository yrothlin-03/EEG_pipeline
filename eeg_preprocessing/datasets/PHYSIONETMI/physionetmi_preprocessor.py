from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import random
import re


MI_RUNS = {4, 6, 8, 10, 12, 14}
BASELINE_RUNS = {1, 2}

RUN_DESC_TO_Y = {
    4: {"T1": 0, "T2": 1},
    8: {"T1": 0, "T2": 1},
    12: {"T1": 0, "T2": 1},

    6: {"T1": 2, "T2": 3},
    10: {"T1": 2, "T2": 3},
    14: {"T1": 2, "T2": 3},
}


class PHYSIONETMI_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = sorted(rootdir.rglob("*.edf"))

        files = [f for f in files if self._get_run_number(f) in MI_RUNS]

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])

        return [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)

    def _get_run_number(self, file_path: Path) -> int | None:
        m = re.match(r"^S\d{3}R(\d{2})\.edf$", file_path.name)
        return int(m.group(1)) if m else None

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = mne.io.read_raw_edf(file_path.as_posix(), preload=False, verbose=False)

        run = self._get_run_number(file_path)
        if run not in RUN_DESC_TO_Y:
            raise ValueError(f"Invalid MI run: {file_path.name}")

        ann = raw.annotations
        onsets, durations, descriptions = [], [], []

        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
            desc = str(desc)

            if desc not in {"T1", "T2"}:
                continue

            label = RUN_DESC_TO_Y[run][desc]

            onsets.append(float(onset))
            durations.append(4.0)
            descriptions.append(str(label))

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=ann.orig_time,
        )

    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^S(\d{3})R\d{2}\.edf$", file_path.name)
        return f"S{m.group(1)}" if m else file_path.stem.split("R")[0]