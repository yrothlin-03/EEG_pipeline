from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import re


BCI2A_STAGE_MAP = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3,
}


class BCI2A_preprocessor(PreprocessorModel):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        super().__init__(dataset_dir, logger)
        self.logger = logger

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        return []

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
        return m.group(1) if m else file_path.stem.split("_")[0]


def test_single_file(file_path: str | Path):
    fp = Path(file_path)
    pre = BCI2A_preprocessor(fp.parent)

    raw = pre.load_data(fp)

    print("File:", fp)
    print("Channels:", raw.ch_names)
    print("Sampling frequency:", raw.info["sfreq"])
    print("Duration (s):", raw.times[-1])
    print("Annotations:")
    for onset, dur, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        print(onset, dur, desc)

    labels = pre.load_labels(fp, raw)

    print("\nSegments after filtering:")
    for i in range(len(labels)):
        print(labels.onset[i], labels.duration[i], labels.description[i])


if __name__ == "__main__":
    test_single_file("/projects/EEG-foundation-model/BCI-IV/A01T.gdf")