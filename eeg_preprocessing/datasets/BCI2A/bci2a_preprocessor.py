from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import random
import re



BCI2A_STAGE_MAP = {
    "769": 0 ,    # left hand
    "770": 1 ,    # right hand
    "771": 2 ,    # foot
    "772": 3 ,    # tongue
}



class BCI2A_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):

        super().__init__(dataset_dir, logger)  
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        filesT = list(rootdir.rglob("A0*T.gdf"))
        # filesE = list(rootdir.rglob("A0*E.gdf"))
        files = filesT
        if self.logger:
            self.logger.info(f"Found {len(files)} GDF files in {rootdir}")

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
        return mne.io.read_raw_gdf(file_path.as_posix(), preload=True, verbose=False)
    
    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = mne.io.read_raw_gdf(file_path.as_posix(), preload=False, verbose=False)

        onsets, durations, descriptions = [], [], []

        for onset, dur, desc in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        ):
            desc = str(desc)
            if desc in BCI2A_STAGE_MAP:
                onsets.append(float(onset) + 1.0)
                durations.append(4.0)
                descriptions.append(str(BCI2A_STAGE_MAP[desc]))

        return mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    
    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^(A0\d)_T\d\.gdf$", file_path.name)
        if m:
            return m.group(1)
        return file_path.stem.split("_")[0]

