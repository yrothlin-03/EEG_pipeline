from datasets.base_model import PreprocessorModel
from pathlib import Path
from logging import Logger
import mne
import random
import re


PATIENT_EPOCHS = {
  'PN00': [
    {'seizure': '1',  'file': 'PN00-1.edf',     'start_sec': 1143,  'end_sec': 1213},
    {'seizure': '2',  'file': 'PN00-2.edf',     'start_sec': 1220,  'end_sec': 1274},
    {'seizure': '3',  'file': 'PN00-3.edf',     'start_sec': 765,   'end_sec': 4425},
    {'seizure': '4',  'file': 'PN00-4.edf',     'start_sec': 1006,  'end_sec': 1080},
    {'seizure': '5',  'file': 'PN00-5.edf',     'start_sec': 904,   'end_sec': 971},
  ],

  'PN01': [
    {'seizure': '1',  'file': 'PN01.edf',       'start_sec': 10218, 'end_sec': 10272},
    {'seizure': '2',  'file': 'PN01.edf',       'start_sec': 46353, 'end_sec': 46427},
  ],

  'PN03': [
    {'seizure': '1',  'file': 'PN03-1.edf',     'start_sec': 38673, 'end_sec': 38784},
    {'seizure': '2',  'file': 'PN03-2.edf',     'start_sec': 34921, 'end_sec': 35054},
  ],

  'PN05': [
    {'seizure': '2',  'file': 'PN05-2.edf',     'start_sec': 7163,  'end_sec': 7198},
    {'seizure': '3',  'file': 'PN05-3.edf',     'start_sec': 6836,  'end_sec': 6866},
    {'seizure': '4',  'file': 'PN05-4.edf',     'start_sec': 3608,  'end_sec': 3647},
  ],

  'PN06': [
    {'seizure': '1',  'file': 'PN06-1.edf',     'start_sec': 5583,  'end_sec': 5647},
    {'seizure': '2',  'file': 'PN06-2.edf',     'start_sec': 8860,  'end_sec': 8929},
    {'seizure': '3',  'file': 'PN06-3.edf',     'start_sec': 6275,  'end_sec': 6317},
    {'seizure': '4',  'file': 'PN06-4.edf',     'start_sec': 5939,  'end_sec': 6002},
    {'seizure': '5',  'file': 'PN06-5.edf',     'start_sec': 4783,  'end_sec': 4827},
  ],

  'PN07': [
    {'seizure': '1',  'file': 'PN07-1.edf',     'start_sec': 22059, 'end_sec': 22121},
  ],

  'PN09': [
    {'seizure': '1',  'file': 'PN09-1.edf',     'start_sec': 7249,  'end_sec': 7329},
    {'seizure': '2',  'file': 'PN09-2.edf',     'start_sec': 7127,  'end_sec': 7186},
    {'seizure': '3',  'file': 'PN09-3.edf',     'start_sec': 7221,  'end_sec': 7285},
  ],

  'PN10': [
    {'seizure': '1A', 'file': 'PN10-1.edf',     'start_sec': 7545,  'end_sec': 7614},
    {'seizure': 'B1', 'file': 'PN10-2.edf',     'start_sec': 7798,  'end_sec': 7849},

    {'seizure': 'B2', 'file': 'PN10-3.edf',
     'clinical_onset_sec': 7835, 'electric_onset_sec': 7841, 'end_sec': 7904},

    {'seizure': 'B3', 'file': 'PN10-4.5.6.edf', 'start_sec': 2309,  'end_sec': 2314},
    {'seizure': 'B4', 'file': 'PN10-4.5.6.edf', 'start_sec': 6544,  'end_sec': 6563},

    {'seizure': 'B5', 'file': 'PN10-4.5.6.edf',
     'clinical_onset_sec': 11225, 'end_sec': 11282},

    {'seizure': 'B7', 'file': 'PN10-7.8.9.edf', 'start_sec': 2748,  'end_sec': 2796},
    {'seizure': 'B8', 'file': 'PN10-7.8.9.edf', 'start_sec': 5459,  'end_sec': 5477},
    {'seizure': 'B9', 'file': 'PN10-7.8.9.edf', 'start_sec': 12923, 'end_sec': 12938},
    {'seizure': 'B10','file': 'PN10-10.edf',    'start_sec': 7977,  'end_sec': 7991},
  ],

  'PN11': [
    {'seizure': '1',  'file': 'PN11.edf',       'start_sec': 7554,  'end_sec': 7609},
  ],

  'PN12': [
    {'seizure': '1',  'file': 'PN12-1.2.edf',   'start_sec': 1312,  'end_sec': 1375},
    {'seizure': '2',  'file': 'PN12-1.2.edf',   'start_sec': 9570,  'end_sec': 9638},
    {'seizure': '3',  'file': 'PN12-3.edf',     'start_sec': 772,   'end_sec': 868},
    {'seizure': '4',  'file': 'PN12-4.edf',     'start_sec': 9812,  'end_sec': 9875},
  ],

  'PN13': [
    {'seizure': '1',  'file': 'PN13-1.edf',     'start_sec': 7062,  'end_sec': 7110},
    {'seizure': '2',  'file': 'PN13-2.edf',     'start_sec': 7249,  'end_sec': 7314},
    {'seizure': '3',  'file': 'PN13-3.edf',     'start_sec': 7553,  'end_sec': 7704},
  ],

  'PN14': [
    {'seizure': '1',  'file': 'PN14-1.edf',     'start_sec': 7262,  'end_sec': 7289},
    {'seizure': '2',  'file': 'PN14-2.edf',     'start_sec': 7479,  'end_sec': 7491},
    {'seizure': '3',  'file': 'PN14-3.edf',     'start_sec': 17540, 'end_sec': 17581},
    {'seizure': '4',  'file': 'PN14-4.edf',     'start_sec': 5463,  'end_sec': 5546},
  ],

  'PN16': [
    {'seizure': '1',  'file': 'PN16-1.edf',     'start_sec': 7184,  'end_sec': 7307},
    {'seizure': '2',  'file': 'PN16-2.edf',     'start_sec': 8574,  'end_sec': 8681},
  ],

  'PN17': [
    {'seizure': '1',  'file': 'PN17-1.edf',     'start_sec': 8420,  'end_sec': 8490},
    {'seizure': '2',  'file': 'PN17-2.edf',     'start_sec': 7731,  'end_sec': 7814},
  ],
}


BASELINE_RUNS = {1, 2}


class SIENA_preprocessor(PreprocessorModel):
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
        kept_files = [f for f in kept_files if self._get_run_number(f) not in BASELINE_RUNS]

        return kept_files

    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        return mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)

    def _get_run_number(self, file_path: Path) -> int | None:
        m = re.match(r"^PN\d{2}-(\d+)\.edf$", file_path.name)
        return int(m.group(1)) if m else None

    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        if raw is None:
            raw = mne.io.read_raw_edf(file_path.as_posix(), preload=False, verbose=False)

        sid = self.get_subject_id(file_path)
        events = [e for e in PATIENT_EPOCHS.get(sid, []) if e.get("file") == file_path.name]

        def _start(e):
            if "start_sec" in e:
                return float(e["start_sec"])
            if "electric_onset_sec" in e:
                return float(e["electric_onset_sec"])
            return float(e["clinical_onset_sec"])

        seizures = []
        for e in events:
            if "end_sec" not in e:
                continue
            try:
                s = _start(e)
            except Exception:
                continue
            t = float(e["end_sec"])
            if t > s:
                seizures.append((s, t))

        seizures.sort()
        merged = []
        for s, t in seizures:
            if not merged or s > merged[-1][1]:
                merged.append([s, t])
            else:
                merged[-1][1] = max(merged[-1][1], t)

        total = float(raw.times[-1]) if raw is not None and raw.n_times > 0 else (merged[-1][1] if merged else 0.0)

        onsets, durations, descriptions = [], [], []
        cur = 0.0
        for s, t in merged:
            s = max(0.0, min(s, total))
            t = max(0.0, min(t, total))
            if s > cur:
                onsets.append(cur)
                durations.append(s - cur)
                descriptions.append("0")
            if t > s:
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


def test_single_file(file_path: str | Path):
    fp = Path(file_path)
    pre = SIENA_preprocessor(fp.parents[1])

    raw = pre.load_data(fp)
    labels = pre.load_labels(fp, raw)

    for i in range(len(labels)):
        print(labels.onset[i], labels.duration[i], labels.description[i])


if __name__ == "__main__":
    test_single_file("/projects/EEG-foundation-model/Siena-scalp/physionet.org/files/siena-scalp-eeg/1.0.0/PN17/PN17-2.edf")