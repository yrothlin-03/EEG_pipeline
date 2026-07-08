# EEG Preprocessing Pipeline
**Telecom Paris – EEG Foundation Model**

Available datasets supported by the preprocessing pipeline.
Each one has a dedicated preprocessor registered in [`datasets/__init__.py`](eeg_preprocessing/datasets/__init__.py).

For any question, issue or suggestion, feel free to contact **@yrothlin-24**.

---

## Available datasets

- **TUEG** (pretraining / SSL) : Temple University Hospital EEG Corpus, large-scale clinical EEG used for self-supervised pretraining (TUAB and TUAR subsets excluded).
- **TUAB** (pathology detection) : TUH Abnormal EEG Corpus, recordings labeled as normal or abnormal.
- **Sleep-EDF** (sleep staging) : polysomnography recordings labeled with sleep stages.
- **CHB-MIT** (seizure detection) : pediatric scalp EEG from Boston Children's Hospital.
- **Siena** (seizure detection) : Siena Scalp EEG Database, adult epilepsy recordings.
- **BCI IV-2a** (motor imagery) : BCI Competition IV dataset 2a, imagined limb movements.
- **PhysioNet MI** (motor imagery) : PhysioNet EEG Motor Movement/Imagery dataset.
- **SHU-MI** (motor imagery) : Shanghai University motor imagery dataset (32 channels).
- **SEED-V** (emotion recognition) : emotional responses elicited from movie clips.
- **FACED** (emotion recognition) : finer-grained affective EEG dataset.
- **BCI 2020 (Track 3)** (imagined speech) : BCI Competition 2020 imagined-speech track.
- **KaraOne** (imagined speech) : imagined speech of phonemes and words.
- **CHISCO** (imagined speech) : Chinese imagined-speech corpus, with reading and recall phases.
- **METASPEECH** (imagined speech / typing) : EEG recorded during a typing task, epoched around each keystroke.
