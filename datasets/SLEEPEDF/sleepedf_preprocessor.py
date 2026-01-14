from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random
import re


SLEEPEDFX_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}


class SLEEPEDF_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = list(rootdir.rglob("*-PSG.edf"))

        if self.logger:
            self.logger.info(f"Found {len(files)} PSG EDF files in {rootdir}")

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
    
    def _get_hypno_file(self, psg_file_path: Path) -> Path:
        if not psg_file_path.name.endswith("-PSG.edf"):
            raise ValueError(f"Not a PSG EDF file: {psg_file_path}")

        base = psg_file_path.name.replace("-PSG.edf", "")  
        prefix = base[:-1]                                 

        matches = sorted(psg_file_path.parent.glob(f"{prefix}*-Hypnogram.edf"))

        if len(matches) == 1:
            return matches[0]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No hypnogram found for {psg_file_path.name} "
                f"(pattern: {prefix}*-Hypnogram.edf)"
            )
        raise RuntimeError(
            f"Multiple hypnograms found for {psg_file_path.name}: "
            f"{[m.name for m in matches]}"
        )

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        hyp = self._get_hypno_file(file_path)
        ann = mne.read_annotations(hyp.as_posix())

        onsets, durations, descriptions = [], [], []

        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
            desc = str(desc)

            if desc not in SLEEPEDFX_STAGE_MAP:
                class_id = -1
            else:
                class_id = SLEEPEDFX_STAGE_MAP[desc]

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
        m = re.match(r"^(SC|ST)(\d+)[A-Z]\d-PSG\.edf$", file_path.name)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        return file_path.stem.split("-")[0]

