from datasets import (PreprocessorModel, 
        TUEG_preprocessor, 
        TUAB_preprocessor, 
        SLEEPEDF_preprocessor, 
        SEEDV_preprocessor, 
        PHYSIONETMI_preprocessor, 
        FACED_preprocessor, 
        BCI2A_preprocessor, 
        SIENA_preprocessor,
        CHBMIT_preprocessor,
        SHUMI_preprocessor  
)
from pathlib import Path
from utils import init_logger, open_out_file
from typing import Dict
from logging import Logger
import mne
import numpy as np
from typing import Any, Iterator, List, Tuple, Dict
import lmdb
import pickle
from typing import Literal
from contextlib import contextmanager
import time
from tqdm import tqdm




def extract_labeled_segments_idx(ann: mne.Annotations, sfreq: float, min_len_sec: float = 0.0):
    out = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        # print(f"Segment duration : {dur} seconds")
        try:
            lab = int(desc)
        except (TypeError, ValueError):
            continue

        if lab == -1 or dur < min_len_sec:
            continue

        s = int(round(onset * sfreq))
        e = int(round((onset + dur) * sfreq))
        if e <= s:
            continue

        out.append((s, e, lab))
    return out


LastMode = Literal["drop", "pad_zero", "repeat", "last"]
WindowMode = Literal["center", "none"]

def extract_windows_from_array(
    x: np.ndarray,
    sfreq: float,
    window_size_sec: float,
    overlap_sec: float,
    window_mode: WindowMode = "center",
    last: LastMode = "last",
) -> Iterator[np.ndarray]:
    if window_size_sec <= 0:
        raise ValueError("window_size_sec must be > 0")
    if overlap_sec < 0 or overlap_sec >= window_size_sec:
        raise ValueError("overlap_sec must be in [0, window_size_sec)")
    if window_mode not in ("center", "none"):
        raise ValueError("window_mode must be 'center' or 'none'")

    n_win = int(round(window_size_sec * sfreq))
    n_step = int(round((window_size_sec - overlap_sec) * sfreq))
    n_step = max(1, n_step)

    n_times = x.shape[1]
    if n_times == 0:
        return

    x = x.astype(np.float32, copy=False)

    if n_times < n_win:
        if window_mode == "none":
            m = int(sfreq)
            if m <= 0:
                raise ValueError("sfreq must be > 0")
            n = (n_times // m) * m
            if n <= 0:
                return
            yield x[:, :n]
            return

        half = n_win / 2.0
        center = int(round(n_times / 2.0))
        s = center - half
        e = center + half

        pad_left = int(max(0, -s))
        pad_right = int(max(0, e - n_times))

        s_clipped = int(max(0, s))
        e_clipped = int(min(n_times, e))

        w = x[:, s_clipped:e_clipped]
        if pad_left > 0 or pad_right > 0:
            w = np.pad(w, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)
        yield w[:, :n_win]
        return

    last_full_start = n_times - n_win
    starts = list(range(0, last_full_start + 1, n_step))

    last_start = starts[-1] if len(starts) > 0 else 0
    remainder = n_times - (last_start + n_win)
    allow_extra = remainder >= (n_win / 2.0)

    if allow_extra and last == "last" and (len(starts) == 0 or starts[-1] != last_full_start):
        starts.append(last_full_start)

    for s in starts:
        yield x[:, s:s + n_win]

    if allow_extra and last in ("pad_zero", "repeat"):
        s = starts[-1] + n_step if len(starts) > 0 else 0
        if s < n_times:
            w = x[:, s:n_times]
            missing = n_win - w.shape[1]
            if missing > 0:
                if last == "pad_zero":
                    pad = np.zeros((w.shape[0], missing), dtype=w.dtype)
                    yield np.concatenate([w, pad], axis=1)
                else:
                    tail = w[:, -1:]
                    pad = np.repeat(tail, missing, axis=1)
                    yield np.concatenate([w, pad], axis=1)



def preprocess_one_file(
    dataset_name: str,
    out_file_env: lmdb.Environment,
    preprocessor: PreprocessorModel,
    file_path: Path,
    logger: Logger = None,
    debug: bool = False,
    log_step: int = 20,
    step_idx: int = 0,
    resample_rate: int = 256,
    bandpass: bool = False,
    l_freq: float = 5.0,
    h_freq: float = 75.0,
    notch: float = 60.0,
    max_amp: float = 300.0,
    trim: float = 0.0,
    min_len_sec: float = 30.0,
    use_iir: bool = True,
    normalize: bool = True,
    window_size_sec: float = 30.0,
    overlap_sec: float = 10.0, 
    global_window_idx: int = 0,
    sub2range: Dict[str, List[Tuple[int, int]]] | None = None,
) -> Tuple[int, Dict[str, Any], int, Dict[str, List[Tuple[int,int]]]]:

    logger.info(f"Preprocessing parameters : \n resample_rate={resample_rate} | bandpass={bandpass} | l_freq={l_freq} | h_freq={h_freq} | notch={notch} | max_amp={max_amp} | trim={trim} | min_len_sec={min_len_sec} | use_iir={use_iir} | normalize={normalize} | window_size_sec={window_size_sec} | overlap_sec={overlap_sec} ")
    raw, y, subject_id = preprocessor.load(file_path)
    # logger.info("Annotations summary:")
    # logger.info(f"  n_annotations = {len(y)}")
    # logger.info(f"  onsets       = {y.onset}")
    # logger.info(f"  durations    = {y.duration}")
    # logger.info(f"  descriptions = {y.description}")

    if resample_rate is not None:
        raw.resample(resample_rate, npad="auto", verbose=False)
    
    if use_iir:
        filter_method = "iir"
    else:
        filter_method = "fir"

    if bandpass:
        raw.filter(l_freq, h_freq, fir_design="firwin", method=filter_method, verbose=False)

    if notch > 0.0:
        raw.notch_filter(notch, fir_design="firwin", method=filter_method, verbose=False)
    
    x = raw.get_data().astype(np.float32)

    if normalize:
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        x = (x - mean) / std
    
    # I don't think trimming is a good idea and needs to be implemented

    raw._data = x

    sfreq = float(raw.info["sfreq"])
    ch_names = raw.info["ch_names"]

    if dataset_name != "TUEG":
        seg_idx = extract_labeled_segments_idx(y, sfreq, min_len_sec=min_len_sec)
    else:
        seg_idx = [(0, x.shape[1], -1)]

    if logger and step_idx == 1:
        logger.info(f"Extracted {len(seg_idx)} labeled segments from file.")

    label_counts = {}
    n_win_total = 0
    start = global_window_idx
    with out_file_env.begin(write=True) as txn:
        for i, (s, e, y_label) in enumerate(seg_idx):
            x_seg = x[:, s:e] 
            win_count = 0
            for x_w in extract_windows_from_array(x_seg, sfreq=sfreq, window_size_sec=window_size_sec, overlap_sec=overlap_sec, last="last"):
                if win_count == 0 and (i == 0 or i%50 == 0):
                    logger.info(f"Window shape: {x_w.shape}")
                label_counts[y_label] = label_counts.get(y_label, 0) + 1
                win_count += 1
                n_win_total += 1

                key = global_window_idx.to_bytes(8, "big")
                record = {
                    "subject_id": subject_id,
                    "x": x_w,
                    "sfreq": sfreq,
                    "ch_names": ch_names,
                    "label": y_label,
                }
                txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                global_window_idx += 1

            if logger and step_idx == 1:
                logger.info(f"  Segment {i:03d}: label={y_label}, n_times={e - s}, extracted {win_count} windows.")

    end = global_window_idx
    if sub2range is not None and end > start:
        sub2range.setdefault(subject_id, []).append((start, end))

    if logger:
        logger.info(f"Finished processing file: {file_path} | Extracted {n_win_total} windows | label distribution: {label_counts}")
    
    return n_win_total, global_window_idx, sub2range


def get_preprocessor(
    dataset_name: str,
    dataset_dir: str,
    logger: Logger = None,
    ) -> PreprocessorModel:
    if dataset_name == "TUEG":
        preprocessor = TUEG_preprocessor(
            dataset_dir,
            logger=logger,
            exclude=["TUAB", "TUAR"],
        )
    elif dataset_name == "TUAB":
        preprocessor = TUAB_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "SLEEPEDF":
        preprocessor = SLEEPEDF_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "SEEDV":
        preprocessor = SEEDV_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "PHYSIONETMI":
        preprocessor = PHYSIONETMI_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "FACED":
        preprocessor = FACED_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "BCI2A":
        preprocessor = BCI2A_preprocessor(
            dataset_dir,
            logger=logger,
        )   
    elif dataset_name == "SIENA":
        preprocessor = SIENA_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "CHBMIT":
        preprocessor = CHBMIT_preprocessor(
            dataset_dir,
            logger=logger,
        )
    elif dataset_name == "SHUMI":
        preprocessor = SHUMI_preprocessor(
            dataset_dir,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return preprocessor


def preprocess_dataset(
    dataset_name: str,
    dataset_dir: str,
    out_dir: str,
    files_ratio: float,
    logger: Logger = None,
    debug: bool  = False,
    log_step: int = 20,
    preprocessing_config: Dict[str, Any] = {}
    ) -> None:

    preprocessor = get_preprocessor(
        dataset_name,
        dataset_dir,
        logger=logger,
    )

    files = preprocessor.get_files(
        ratio=files_ratio,
        seed=preprocessing_config.get("seed", 42),
    )

    out_file_env, out_file_path = open_out_file(out_dir, dataset_name, file_format="lmdb")
    if logger:
        logger.info(f"Opened output LMDB file at {out_file_path} | map_size={out_file_env.info()['map_size']/(1024**3):.2f} GiB")

    total_windows = 0
    all_subj_ids = set()
    sub2range = {}

    for idx, file_path in enumerate(tqdm(files, desc="Processing files")):
        n_win, total_windows, sub2range = preprocess_one_file(dataset_name, out_file_env, preprocessor, Path(file_path), logger=logger, log_step=log_step, debug=debug, step_idx=idx+1, global_window_idx=total_windows, sub2range=sub2range, **preprocessing_config)
        subj_id = preprocessor.get_subject_id(Path(file_path))
        all_subj_ids.add(subj_id)
        if idx%log_step == 0:
            logger.info(f"File {idx+1}: {file_path} | Subject ID: {subj_id} | Extracted {n_win} windows. Total windows so far: {total_windows}")


    with out_file_env.begin(write=True) as txn:
        keys_list = sorted(list(all_subj_ids))
        keys_data = pickle.dumps(keys_list)
        n_subj = len(keys_list)
        txn.put(b"__subject_ids__", keys_data)
        txn.put(b"__len__", pickle.dumps(total_windows))
        txn.put(b"__subj_ranges__", pickle.dumps(sub2range))

    if logger:
        logger.info(f"Total unique subjects processed: {len(all_subj_ids)}")
        logger.info("Subject ID to ranges mapping (showing up to 20 subjects):")
    for i, (k, L) in enumerate(sub2range.items()):
        logger.info(f"  Subject ID: {k} | Ranges: {L}")
        if i >= 20:
            logger.info("  ...")
            break

    if logger:
        logger.info(f"[END] Preprocessing completed. Total windows extracted: {total_windows}")




# preprocessing_params = {
#     "resample_rate": 256,
#     "l_freq": 0.5,
#     "h_freq": 75.0,
#     "notch": 60.0,
#     "normalize": True,
#     "bandpass": False,
#     "max_amp": 300.0,  # in muV
#     "trim": 0.0,
#     "min_len_sec": 1.0,
#     "window_size_sec": 30.0,
#     "overlap_sec": 0.0,
#     "use_iir": True
# }

# if __name__ == "__main__":
#     dataset_name = "TUEG"
#     dataset_dir = "/projects/EEG-foundation-model/tuh_eeg/v2.0.1/edf/"
#     out_dir = "/projects/EEG-foundation-model/RECH202/tests_preprocessing/"
#     log_path = "/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/logs/tests/preprocessing.log"
#     files_ratio = 0.01
#     debug = True
#     logger = init_logger(log_path)
#     log_step = 20
#     preprocess_dataset(
#         dataset_name,
#         dataset_dir,
#         out_dir,
#         files_ratio,
#         logger=logger,
#         debug=debug,
#         log_step=log_step,
#         preprocessing_config=preprocessing_params,
#     )

