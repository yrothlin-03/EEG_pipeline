# EEG Preprocessing Pipeline  
**Telecom Paris – EEG Foundation Model**

## Presentation

This repository contains the EEG preprocessing pipeline developed for the  
**EEG Foundation Model project at Telecom Paris**. The main idea is to have a pipeline that is reproducible and scalable to new datasets but in order to register a new dataset you have to work a bit so that the information preprocessed are correct.

The goal of this pipeline is to provide a **unified, reproducible, and scalable**
preprocessing framework for heterogeneous EEG datasets, supporting both:

- **Self-supervised / pretraining datasets** (e.g. TUEG \wo TUAB, TUAR)
- **Supervised classification datasets** (e.g. TUAB, Sleep-EDF, BCI, FACED, SEEDV, PhysioNet MI)

If you have any questions, issues, or suggestions, feel free to contact  
**@yrothlin-24**.

---

## 1. Pipeline Overview

<!-- ### 1.1 Design principles
- Dataset-agnostic preprocessing interface
- Single-pass raw loading (no redundant I/O)
- Explicit label semantics (`ignore`, `unlabeled`, `supervised`)
- Scalable storage using **LMDB**
- Compatibility with **MNE** and **PyTorch DataLoader** -->

### 1.2 Supported datasets
- TUEG (pretraining / unlabeled) (without TUAB and TUAR files)
- TUAB (normal vs abnormal)
- Sleep-EDF
- PhysioNet Motor Imagery
- BCI Competition IV-2a
- FACED
- SEED-V

---

## 2. Preprocessing Method

### 2.1 Signal preprocessing
- Resampling
- Band-pass filtering
- Notch filtering
- Normalization (per-channel z-score)
- Optional trimming (even though the parameter exists, it does not apply because I don't believe in that)

### 2.2 Windowing strategy
- Fixed-length sliding windows (ex: if you have a labeled segment of 100 seconds, you will have three 30 seconds window and for the last one you have several option (padding, last, ignore,…). By default it will take the last 30 seconds of the segment.)
- Configurable window size and overlap
- Handling of short segments (padding / repetition)

### 2.3 Label handling
- Labels are stored as **integers**
- Convention:
  - `label >= 0` : supervised class
  - `label = -1` : ignored segment
  - `label = -2` : unlabeled (pretraining / SSL) (although right now I just filter with tueg name but I will implement that later for generalizability)

Only valid segments are written to the output dataset.

---

## 3. Dataset Interface

### 3.1 `PreprocessorModel` abstraction

Each dataset must implement a dedicated preprocessor inheriting from
`PreprocessorModel`.

Mandatory methods:
- `load_data(file_path)`
- `load_labels(file_path, raw=None)`
- `get_subject_id(file_path)`
- `get_files(ratio, seed)`

<!-- ### 3.2 Single-pass loading
Raw signals are loaded **once per file** and passed to the label constructor
to avoid unnecessary disk access. -->

---

## 4. Adding a New Dataset

### 4.1 Step-by-step guide

1. Create a new preprocessor in `datasets/<DATASET_NAME>/`
2. Inherit from `PreprocessorModel`
3. Implement:
    - Files retrieving
   - Raw loading (`MNE`) you have to handle the format yourself. This way it forces you to check what the files contain to return proper informations.
   - Label remapping to integer classes
   - Subject ID parsing
4. Register the dataset in `get_preprocessor(...)`
5. Add the preprocessing parameters in the configuration file (`configs/preprocessing.yaml`). Basically you just just do the same as for the others (where to get the files, the logging step, the fraction of the dataset you want to preprocess).

<!-- ### 4.2 Label remapping requirement
All datasets must output labels as:
```python
str(int_class_id) -->

--- 
### 5. Starting a preprocessing
In the configuration file :
1. Specify the `dataset_name`.
2. Specify the logging output path `logger_dir`.
3. Specify the out_dir `out_dir`.
You can check the other parameters too.


--- 
### 6. Exemple of use for data loading (pretraining / downstream)

1. Clone the repository in your work repository.
```python
from loaders import build_loaders

train_loader, val_loader, test_loader = build_loaders(
        lmdb_path,  # path to the lmdb file containing your preprocessed dataset, USING THIS PREPROCESSING PIPELINE
        split_ratio=(0.8, 0.1, 0.1),
        batch_size=1,
        seed=42,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        shuffle_val=False,  # or big datasets with uneven labeling, you might to shuffle you val if you want to run some tests.
)

print(f"Train size : {len(train_loader)} | Validation size : {len(val_loader)} | Test size : {len(test_loader)} ")
for step, (x, y) in enumerate(train_loader):
    print(x.shape)
    print(y)
    if step >= 5:
        break
```