from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger, getLogger
import mne
import random


DATASET_PATH = {
    "TUAB": "/projects/EEG-foundation-model/tuh_eeg_abnormal/v3.0.1",
    "TUAR": "/projects/EEG-foundation-model/tuh_eeg_artifact/v3.0.1",
}


class TUEG_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None, exclude: list[str] = None):
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude
        super().__init__(dataset_dir, logger)  
    
    def _get_tuab_files(self, tuab_dir: Path = Path("/projects/EEG-foundation-model/tuh_eeg_abnormal/v3.0.1")) -> set[str]:
        return {f.stem for f in tuab_dir.rglob("*.edf")}

    def _get_tuar_files(self, tuar_dir: Path = Path("/projects/EEG-foundation-model/tuh_eeg_artifact/v3.0.1")) -> set[str]:
        return {f.stem for f in tuar_dir.rglob("*.edf")}

    def _collect_dataset_stats(self, dataset_root: Path) -> tuple[set[str], set[str], int]:
        dataset_root = dataset_root / "edf" if (dataset_root / "edf").exists() else dataset_root
        edf_files = list(dataset_root.rglob("*.edf"))

        stems = {f.stem for f in edf_files}
        subjects = {self.get_subject_id(f) for f in edf_files}

        return stems, subjects, len(edf_files)

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

        excluded_stems: set[str] = set()

        if "TUAB" in self.exclude:
            tuab_root = Path(DATASET_PATH["TUAB"])
            tuab_stems, tuab_subjects, tuab_n_files = self._collect_dataset_stats(tuab_root)
            excluded_stems |= tuab_stems
            if self.logger:
                self.logger.info(
                    f"[TUAB] files={tuab_n_files}, unique_subjects={len(tuab_subjects)}, "
                    f"unique_stems={len(tuab_stems)}"
                )

        if "TUAR" in self.exclude:
            tuar_root = Path(DATASET_PATH["TUAR"])
            tuar_stems, tuar_subjects, tuar_n_files = self._collect_dataset_stats(tuar_root)
            excluded_stems |= tuar_stems
            if self.logger:
                self.logger.info(
                    f"[TUAR] files={tuar_n_files}, unique_subjects={len(tuar_subjects)}, "
                    f"unique_stems={len(tuar_stems)}"
                )

        if excluded_stems:
            before = len(files)
            files = [f for f in files if f.stem not in excluded_stems]
            after = len(files)

            remaining_subjects = {self.get_subject_id(f) for f in files}

            if self.logger:
                self.logger.info(
                    f"Excluded by stem: removed {before-after} files "
                    f"(excluded_stems={len(excluded_stems)}), remaining_files={after}, "
                    f"remaining_unique_subjects={len(remaining_subjects)}"
                )

        if not files:
            raise ValueError("No files found after exclusions")

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
    
    def load_labels(self, file_path: Path) -> mne.Annotations:
        return None
    
    def get_subject_id(self, file_path: Path) -> str:
        return file_path.stem.split("_")[0]



# if __name__ == "__main__":
#     tueg_path = "/projects/EEG-foundation-model/tuh_eeg/v2.0.1/edf"
#     preprocessor = TUEG_preprocessor(tueg_path, logger=logger, exclude=["TUAB", "TUAR"])
#     files = preprocessor.get_files(ratio=0.1, seed=42)
#     print(f"Number of files: {len(files)}")