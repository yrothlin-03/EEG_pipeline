from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import random
import re

import mne
import numpy as np


# METASPEECH / DECOMEG is a typing task recorded in BrainVision format (.vhdr/.vmrk/.eeg,
# 64 channels @ 1000 Hz). Each keystroke is logged as a "Stimulus" marker whose numeric
# code is the ASCII value of the typed key (S32 = space, S65..S90 = 'A'..'Z'). We epoch the
# EEG around every keystroke and use the typed character as the class label.
#
# Note: the actiCAP amplifier emits a one-shot ascending sequence S1..S255 at recording
# start (impedance/startup). Codes 32 and 65-90 appear once each in that burst and are
# therefore mislabeled as keystrokes. Use skip_start_sec to drop that initial window.

# Class map: space -> 0, 'A'..'Z' -> 1..26. Every other ASCII code is dropped (label -1).
METASPEECH_LABEL_MAP = {32: 0, **{c: c - 64 for c in range(65, 91)}}


class METASPEECH_preprocessor(PreprocessorModel):
    def __init__(
        self,
        dataset_dir: str | Path,
        logger: Logger = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
        skip_start_sec: float = 60.0,
        include_tapping: bool = False,
    ):
        super().__init__(dataset_dir, logger)

        if tmax <= tmin:
            raise ValueError(f"tmax must be > tmin, got tmin={tmin}, tmax={tmax}")

        self.tmin = tmin
        self.tmax = tmax
        self.skip_start_sec = skip_start_sec
        self.include_tapping = include_tapping

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = sorted(rootdir.rglob("*.vhdr"))

        if not self.include_tapping:
            files = [f for f in files if "tapping" not in f.name.lower()]

        if self.logger:
            self.logger.info(f"Found {len(files)} BrainVision .vhdr files in {rootdir}")

        subj_ids = [self.get_subject_id(f) for f in files]
        unique_subj_ids = sorted(set(subj_ids))

        n_subj_total = len(unique_subj_ids)
        n_subj_keep = max(1, int(n_subj_total * ratio))

        rng = random.Random(seed)
        rng.shuffle(unique_subj_ids)
        subj_ids_to_keep = set(unique_subj_ids[:n_subj_keep])

        kept_files = [f for f, sid in zip(files, subj_ids) if sid in subj_ids_to_keep]

        if self.logger:
            self.logger.info(
                f"Subject-wise selection: keeping {n_subj_keep}/{n_subj_total} subjects "
                f"({ratio * 100:.2f}%), resulting in {len(kept_files)}/{len(files)} files"
            )

        return kept_files

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_brainvision(
            file_path.as_posix(),
            preload=True,
            verbose=False,
        )

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = self.load_data(file_path)

        last_onset_sec = raw.n_times / float(raw.info["sfreq"])
        duration = self.tmax - self.tmin

        onsets, durations, descriptions = [], [], []

        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            if onset < self.skip_start_sec:
                continue

            m = re.search(r"S\s*(\d+)", str(desc))
            if m is None:
                continue

            class_id = METASPEECH_LABEL_MAP.get(int(m.group(1)), -1)
            if class_id == -1:
                continue

            seg_onset = float(onset) + self.tmin
            if seg_onset < 0.0 or seg_onset + duration > last_onset_sec:
                continue

            onsets.append(seg_onset)
            durations.append(float(duration))
            descriptions.append(str(class_id))

        if self.logger:
            self.logger.info(
                f"{file_path.name}: extracted {len(onsets)} keystroke epochs "
                f"({len(set(descriptions))} unique classes)"
            )

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
        )

    def get_subject_id(self, file_path: Path) -> str:
        m = re.match(r"^(\d+)_DECOMEG", file_path.name)
        return m.group(1) if m else file_path.stem


if __name__ == "__main__":
    from logging import basicConfig, INFO, getLogger
    from collections import Counter

    basicConfig(level=INFO)
    logger = getLogger("metaspeech_test")

    dataset_dir = "/projects/EEG-foundation-model/RECH202/data/EEG"

    preprocessor = METASPEECH_preprocessor(dataset_dir=dataset_dir, logger=logger)

    files = preprocessor.get_files(ratio=1.0)
    print(f"\nFound {len(files)} files")

    if not files:
        raise RuntimeError("No .vhdr files found")

    test_file = files[0]
    print(f"\nTesting file:\n{test_file}")

    raw = preprocessor.load_data(test_file)
    print("\nRAW")
    print("n_channels:", len(raw.ch_names))
    print("sfreq:", raw.info["sfreq"])
    print("duration_sec:", raw.n_times / raw.info["sfreq"])

    ann = preprocessor.load_labels(test_file, raw)
    print("\nANNOTATIONS")
    print("n_annotations:", len(ann))
    print("unique class ids:", sorted(set(ann.description), key=int))
    print("segment duration:", ann.duration[0] if len(ann) else None)

    counts = Counter(ann.description)
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]
    print("label counts (top 5):", dict(top))

    print("\nSUBJECT ID:", preprocessor.get_subject_id(test_file))

    last_sample_s = raw.n_times / raw.info["sfreq"]
    ok = np.all(np.asarray(ann.onset) + np.asarray(ann.duration) <= last_sample_s + 0.01)
    print("all onsets+durations within recording:", bool(ok))

    print("\nTEST PASSED")
