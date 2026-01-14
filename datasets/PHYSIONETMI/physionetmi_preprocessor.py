from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random
import re


PHYSIONETMI_STAGE_MAP = {
    "T0": 0,    # rest
    "T1": 1,    # left hand or both fists (depending on run)
    "T2": 2,    # right hand or both feet (depending on run)
}



class PHYSIONETMI_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = list(rootdir.rglob("*.edf"))
        if self.logger:
            self.logger.info(f"Found {len(files)} EDF files in {rootdir}")

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = int(n_subj_total * ratio)
        n_subj_keep = max(1, n_subj_keep)

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
        return mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)
    
    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        ann = raw.annotations

        onsets, durations, descriptions = [], [], []

        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
            desc = str(desc)

            if desc not in PHYSIONETMI_STAGE_MAP:
                continue

            class_id = PHYSIONETMI_STAGE_MAP[desc]

            onsets.append(float(onset))
            durations.append(float(dur) if float(dur) > 0 else 0.0)
            descriptions.append(str(class_id))  

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=ann.orig_time,
        )
    
    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^S(\d{3})R\d{2}\.edf$", file_path.name)
        if m:
            return f"S{m.group(1)}"
        return file_path.stem.split("R")[0]

