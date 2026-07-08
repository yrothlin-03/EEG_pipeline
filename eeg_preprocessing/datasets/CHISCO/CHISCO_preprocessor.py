from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import json
import random
import re
from typing import Literal

import mne
import numpy as np
import pandas as pd


# Trial structure (paper Fig. 3):
#   t=0s    : trigger '100' — Reading Phase starts (participant reads the sentence)
#   t=5.0s  : Recall Phase starts (participant imagines speaking the sentence)
#   t=8.3s  : Recall Phase ends — Rest Interval starts
#   t=10.1s : Rest Interval ends (next trial)
#
# For imagined speech decoding, the relevant EEG is the Recall Phase (5.0s → 8.3s).
# Use segment="recall" (default) for BCI/decoding tasks.
# Use segment="full" if the pipeline's window_size_sec > 3.3s and you want more data.

_SEGMENT_OFFSETS: dict[str, tuple[float, float]] = {
    #              (onset_offset_s, duration_s)
    "reading": (0.0, 5.0),   # reading phase only
    "recall":  (5.0, 3.3),   # imagined speech phase only — use window_size_sec ≤ 3.3
    "full":    (0.0, 8.3),   # reading + recall (no rest) — use window_size_sec ≤ 8.3
}

SegmentType = Literal["reading", "recall", "full"]

CHISCO_CH_TYPE_MAP = {
    "VEO": "eog",
    "HEO": "eog",
    "EKG": "ecg",
    "EMG": "emg",
    "Trigger": "stim",
}


class CHISCO_preprocessor(PreprocessorModel):
    def __init__(
        self,
        dataset_dir: str | Path,
        logger: Logger = None,
        segment: SegmentType = "recall",
    ):
        super().__init__(dataset_dir, logger)

        if segment not in _SEGMENT_OFFSETS:
            raise ValueError(
                f"segment must be one of {list(_SEGMENT_OFFSETS)}, got {segment!r}"
            )
        self.segment = segment
        self._onset_offset_s, self._duration_s = _SEGMENT_OFFSETS[segment]

        textmaps_path = self.dataset_dir / "json" / "textmaps.json"
        if not textmaps_path.exists():
            raise FileNotFoundError(f"textmaps.json not found: {textmaps_path}")

        with open(textmaps_path) as f:
            self._text_map: dict[str, int] = json.load(f)

    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        rootdir = Path(self.dataset_dir).expanduser()
        files = sorted(rootdir.rglob("sub-*/ses-*/eeg/*.edf"))

        if self.logger:
            self.logger.info(f"Found {len(files)} EDF files in {rootdir}")

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
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        rename = {ch: t for ch, t in CHISCO_CH_TYPE_MAP.items() if ch in raw.ch_names}
        raw.set_channel_types(rename)
        return raw

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        run_num = self._run_number(file_path)
        if run_num is None:
            raise ValueError(f"Cannot extract run number from filename: {file_path.name}")

        xlsx_path = self.dataset_dir / "textdataset" / f"split_data_{run_num}.xlsx"
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Text data not found: {xlsx_path}")

        df = pd.read_excel(xlsx_path, usecols=[0], header=0)
        sentences: list[str] = df.iloc[:, 0].tolist()

        if raw is None:
            raw = self.load_data(file_path)

        # Trigger '100' marks the start of the Reading Phase (t=0 of each trial).
        # We shift the onset by self._onset_offset_s to select the desired phase.
        trigger_onsets = raw.annotations.onset[raw.annotations.description == "100"]

        n = min(len(trigger_onsets), len(sentences))

        if self.logger and n < max(len(trigger_onsets), len(sentences)):
            self.logger.warning(
                f"{file_path.name}: trial count mismatch — "
                f"{len(trigger_onsets)} EDF triggers vs {len(sentences)} xlsx rows, "
                f"keeping {n}"
            )

        onsets = trigger_onsets[:n] + self._onset_offset_s
        durations = np.full(n, self._duration_s, dtype=float)
        descriptions = [
            str(self._text_map.get(sentences[i], -1))
            for i in range(n)
        ]

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=None,
        )

    def get_subject_id(self, file_path: Path) -> str:
        for part in reversed(file_path.parts):
            if re.match(r"^sub-\d+$", part):
                return part
        return file_path.parts[-4]

    @staticmethod
    def _run_number(file_path: Path) -> int | None:
        m = re.search(r"run-(\d+)", file_path.name)
        return int(m.group(1)) if m else None


if __name__ == "__main__":
    from logging import basicConfig, INFO, getLogger

    basicConfig(level=INFO)
    logger = getLogger("chisco_test")

    dataset_dir = (
        "/projects/EEG-foundation-model/RECH202/speech_raw_datasets/ds005170"
    )

    for seg in ("reading", "recall", "full"):
        print(f"\n{'='*60}")
        print(f"segment={seg!r}")
        preprocessor = CHISCO_preprocessor(dataset_dir=dataset_dir, logger=logger, segment=seg)

        files = preprocessor.get_files(ratio=1.0)
        test_file = files[0]

        raw = preprocessor.load_data(test_file)
        ann = preprocessor.load_labels(test_file, raw)

        sfreq = raw.info["sfreq"]
        print(f"n_annotations: {len(ann)}")
        print(f"segment duration: {ann.duration[0]:.1f}s")
        print(f"First trial onset: {ann.onset[0]:.3f}s")

        # verify onset is within recording
        last_sample_s = raw.n_times / sfreq
        onsets_ok = np.all(ann.onset + ann.duration <= last_sample_s + 0.01)
        print(f"All onsets+durations within recording: {onsets_ok}")

        print(f"label counts (top 5): {dict(list(sorted(Counter(ann.description).items(), key=lambda x: -x[1])[:5]))}")

    print("\nTEST PASSED")
