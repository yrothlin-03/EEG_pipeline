from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import random
from .const import RECORDS_WITH_SEIZURES, SEIZURES


class CHBMIT_preprocessor(PreprocessorModel):
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
        self.logger.info(f"Total subjects: {n_subj_total} | Kept subjects: {n_subj_keep} | Total files : {len(files)} | Kept files: {len(kept_files)}")
        return kept_files

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = mne.io.read_raw_edf(file_path.as_posix(), preload=False, verbose=False)

        fname = file_path.name
        total = float(raw.times[-1]) if raw.n_times > 0 else 0.0

        seizures = []
        for s, t in SEIZURES.get(fname, []):
            s = float(s)
            t = float(t)
            if t > s:
                seizures.append((s, t))

        seizures.sort()
        merged = []
        for s, t in seizures:
            s = max(0.0, min(s, total))
            t = max(0.0, min(t, total))
            if not merged or s > merged[-1][1]:
                merged.append([s, t])
            else:
                merged[-1][1] = max(merged[-1][1], t)

        onsets, durations, descriptions = [], [], []
        cur = 0.0
        for s, t in merged:
            if s > cur:
                onsets.append(cur)
                durations.append(s - cur)
                descriptions.append("0")
            onsets.append(s)
            durations.append(t - s)
            descriptions.append("1")
            cur = max(cur, t)

        if total > cur:
            onsets.append(cur)
            durations.append(total - cur)
            descriptions.append("0")

        ann = raw.annotations
        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=ann.orig_time,
        )

    def get_subject_id(self, file_path: Path) -> str:
        return file_path.parent.name