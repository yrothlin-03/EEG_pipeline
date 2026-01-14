from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random

TUAB_STAGE_MAP = {
    "normal": 0,
    "abnormal": 1,
}



class TUAB_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        root = (
            self.dataset_dir / "edf"
            if (self.dataset_dir / "edf").exists()
            else self.dataset_dir
        )

        files = list(root.rglob("*.edf"))

        if self.logger:
            self.logger.info(f"Found {len(files)} EDF files in {root}")

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
    
    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw) -> mne.Annotations:
        parts = [p.lower() for p in file_path.parts]

        if "normal" in parts:
            class_id = TUAB_STAGE_MAP["normal"]   
        elif "abnormal" in parts:
            class_id = TUAB_STAGE_MAP["abnormal"] 
        else:
            raise ValueError(f"Cannot infer TUAB label from path parts: {file_path}")

        onset = 0.0
        duration = float(raw.times[-1])  

        return mne.Annotations(
            onset=[onset],
            duration=[duration],
            description=[str(class_id)],  
            orig_time=None,
        )
        
    def get_subject_id(self, file_path: Path) -> str:
        return file_path.stem.split("_")[0]

