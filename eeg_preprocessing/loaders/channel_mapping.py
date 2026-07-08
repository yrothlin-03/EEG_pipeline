import re
from pathlib import Path
from typing import List, Literal
import torch


EEG_LIKE_SET = {
    "FP1","FPZ","FP2",
    "AF7","AF3","AFZ","AF4","AF8",
    "F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FT8","FT9","FT10",
    "FC5","FC3","FC1","FCZ","FC2","FC4","FC6",
    "T7","T8","T9","T10","T3","T4","T5","T6",
    "C5","C3","C1","CZ","C2","C4","C6",
    "TP7","TP8",
    "CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8",
    "O1","OZ","O2",
    "IZ",
    "A1","A2"
}

_NON_EEG_PAT = re.compile(
    r"(EOG|EMG|ECG|EKG|PPG|PLETH|RESP|RESPIR|AIRFLOW|NASAL|THOR|ABDO|"
    r"SNORE|MIC|AUX|TRIG|TRIGGER|EVENT|MARK|REF|GND|GROUND|"
    r"M1|M2|EAR|MASTOID|CHIN|LEG|ARM|CHEST|BELT|TEMP|HEART|PULSE|"
    r"SaO2|SPO2|CO2|ETCO2|GSR|EDA|ACC|GYRO|MAG|PHOTO|PHOTIC)",
    re.IGNORECASE
)


_PREFIX_PAT = re.compile(r"^\s*(EEG|MEG|ECG|EKG)\s*[-:]?\s*", re.IGNORECASE)
_SUFFIX_PAT = re.compile(r"\s*[-:]?\s*(REF|LE|RE)\s*$", re.IGNORECASE)
_JUNK_PAT = re.compile(r"[.\s_]+")
_KEEP_PAT = re.compile(r"[^A-Z0-9\-]+")



TARGET_CHS = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","FZ","CZ","PZ"]


TUEG_MAPPING = {
    "FP1": "FP1", "FP2": "FP2",
    "F7": "F7", "F3": "F3", "FZ": "FZ", "F4": "F4", "F8": "F8",
    "T3": "T7", "T7": "T7",
    "C3": "C3", "CZ": "CZ", "C4": "C4",
    "T4": "T8", "T8": "T8",
    "T5": "P7", "P7": "P7",
    "P3": "P3", "PZ": "PZ", "P4": "P4",
    "T6": "P8", "P8": "P8",
    "O1": "O1", "O2": "O2",
}

TUAB_MAPPING = TUEG_MAPPING.copy()

PHYSIONET_MAPPING = {
    **{k: k for k in TARGET_CHS},
    "FPZ": "FZ", "FCZ": "FZ", "CPZ": "PZ", "POZ": "PZ",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

SLEEPEDF_MAPPING = {
    "FPZ": "FZ",
    "CZ":  "CZ",
    "PZ":  "PZ",
    "OZ":  "O1",
}

SEEDV_MAPPING = {
    **{k: k for k in TARGET_CHS},
    "FPZ": "FZ", "AFZ": "FZ", "FCZ": "FZ", "CPZ": "PZ", "POZ": "PZ",
    "AF3": "F3", "AF4": "F4",
    "F1": "FZ", "F2": "FZ",
    "FC1": "F3", "FC2": "F4", "FC3": "F3", "FC4": "F4", "FC5": "F7", "FC6": "F8",
    "C1": "CZ", "C2": "CZ", "C5": "C3", "C6": "C4",
    "CP1": "P3", "CP2": "P4", "CP3": "P3", "CP4": "P4", "CP5": "P7", "CP6": "P8",
    "P1": "PZ", "P2": "PZ", "P5": "P3", "P6": "P4",
    "PO3": "O1", "PO4": "O2", "PO5": "O1", "PO6": "O2", "PO7": "O1", "PO8": "O2",
    "OZ": "O1",
}

FACED_MAPPING = {
    'FP1': 'FP1', 'FP2': 'FP2', 'FZ': 'FZ', 'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
    'FC1': 'FC1', 'FC2': 'FC2', 'FC5': 'FC5', 'FC6': 'FC6',
    'CZ': 'CZ', 'C3': 'C3', 'C4': 'C4',
    'T7': 'T7', 'T8': 'T8',
    'CP1': 'CP1', 'CP2': 'CP2', 'CP5': 'CP5', 'CP6': 'CP6',
    'PZ': 'PZ', 'P3': 'P3', 'P4': 'P4', 'P7': 'P7', 'P8': 'P8',
    'PO3': 'PO3', 'PO4': 'PO4', 'OZ': 'OZ', 'O1': 'O1', 'O2': 'O2'
}

BCI2A_MAPPING = {
    "FZ": "FZ",
    "C3": "C3",
    "CZ": "CZ",
    "C4": "C4",
    "PZ": "PZ",

    "0":  "F3",
    "1":  "F3",
    "2":  "FZ",
    "3":  "F4",
    "4":  "F4",

    "5":  "C3",
    "6":  "C3",
    "7":  "C4",
    "8":  "C4",

    "9":  "P3",
    "10": "P3",
    "11": "PZ",
    "12": "P4",
    "13": "P4",

    "14": "P3",
    "15": "P4",
    "16": "PZ",
}

SIENA_MAPPING = {
    "FP1": "FP1", "FP2": "FP2",
    "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8", "FZ": "FZ",
    "C3": "C3", "C4": "C4", "CZ": "CZ",
    "P3": "P3", "P4": "P4", "PZ": "PZ",
    "O1": "O1", "O2": "O2",
    "FC1": "F3", "FC2": "F4", "FC5": "F7", "FC6": "F8",
    "CP1": "P3", "CP2": "P4", "CP5": "P7", "CP6": "P8",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "F9": "F7", "F10": "F8",
}

SHUMI_MAPPING = {
    "FP1": "FP1", "FP2": "FP2",
    "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8", "FZ": "FZ",
    "C3": "C3", "C4": "C4", "CZ": "CZ",
    "P3": "P3", "P4": "P4", "PZ": "PZ",
    "O1": "O1", "O2": "O2", "OZ": "O1",
    "PO3": "O1", "PO4": "O2",
    "FC1": "F3", "FC2": "F4", "FC5": "F7", "FC6": "F8",
    "CP1": "P3", "CP2": "P4", "CP5": "P7", "CP6": "P8",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

FACED_CHANNEL_LIST = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4',
    'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

FACED_LOCATION_LIST = [['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
                       ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
                       ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
                       ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
                       ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
                       ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
                       ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
                       ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']]

FACED_ADJACENCY_LIST = {
    'FP1': ['F3', 'FZ'],
    'FP2': ['FZ', 'F4'],
    'F7': ['FC5'],
    'F3': ['FC1', 'FC5'],
    'FZ': ['AF4', 'FC2', 'FC1'],
    'F4': ['AF4', 'FC6', 'FC2'],
    'F8': ['FC6'],
    'FC5': ['F7', 'F3', 'C3', 'T7'],
    'FC1': ['F3', 'FZ', 'CZ', 'C3'],
    'FC2': ['FZ', 'F4', 'C4', 'CZ'],
    'FC6': ['F4', 'F8', 'T8', 'C4'],
    'T7': ['FC5', 'CP5'],
    'C3': ['FC5', 'FC1', 'CP1', 'CP5'],
    'CZ': ['FC1', 'FC2', 'CP2', 'CP1'],
    'C4': ['FC2', 'FC6', 'CP6', 'CP2'],
    'T8': ['FC6', 'CP6'],
    'CP5': ['T7', 'C3', 'P3', 'P7'],
    'CP1': ['C3', 'CZ', 'PZ', 'P3'],
    'CP2': ['CZ', 'C4', 'P4', 'PZ'],
    'CP6': ['C4', 'T8', 'P8', 'P4'],
    'P7': ['CP5'],
    'P3': ['CP5', 'CP1', 'PO3'],
    'PZ': ['CP1', 'CP2', 'PO4', 'PO3'],
    'P4': ['CP2', 'CP6', 'PO4'],
    'P8': ['CP6'],
    'PO3': ['P3', 'PZ', 'OZ', 'O1'],
    'PO4': ['PZ', 'P4', 'O2', 'OZ'],
    'O1': ['PO3', 'OZ'],
    'OZ': ['PO3', 'PO4', 'O2', 'O1'],
    'O2': ['PO4', 'OZ']
}



CHBMIT_MAPPING = {
    "FP1-F7":   "FP1",
    "FP1-F3":   "FP1",
    "FP2-F4":   "FP2",
    "FP2-F8":   "FP2",

    "F7-T7":    "F7",
    "F8-T8":    "F8",

    "F3-C3":    "F3",
    "F4-C4":    "F4",

    "T7-P7":    "T7",
    "P7-T7":    "T7",
    "T8-P8-0":  "T8",
    "T8-P8-1":  "T8",

    "C3-P3":    "C3",
    "C4-P4":    "C4",

    "C3-P3":    "C3",
    "C4-P4":    "C4",

    "P7-O1":    "P7",
    "P8-O2":    "P8",

    "C3-P3":    "C3",
    "C4-P4":    "C4",

    "P3-O1":    "P3",
    "P4-O2":    "P4",

    "CZ-PZ":    "CZ",
    "FZ-CZ":    "FZ",

    "T7-FT9":   "T7",
    "FT10-T8":  "T8",
    "FT9-FT10": "T7",
    "FT10-T8":  "T8",
}

KARAONE_MAPPING = {
    "FP1": "FP1",
    "FP2": "FP2",

    "F3": "F3",
    "F4": "F4",
    "F7": "F7",
    "F8": "F8",
    "FZ": "FZ",

    "C3": "C3",
    "C4": "C4",
    "CZ": "CZ",

    "P3": "P3",
    "P4": "P4",
    "P7": "P7",
    "P8": "P8",
    "PZ": "PZ",

    "O1": "O1",
    "O2": "O2",

    "T7": "T7",
    "T8": "T8",

    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",

    "FC5": "F7",
    "FC3": "F3",
    "FC1": "F3",
    "FCZ": "FZ",
    "FC2": "F4",
    "FC4": "F4",
    "FC6": "F8",

    "CP5": "P7",
    "CP3": "P3",
    "CP1": "P3",
    "CPZ": "PZ",
    "CP2": "P4",
    "CP4": "P4",
    "CP6": "P8",

    "PO7": "O1",
    "PO3": "O1",
    "POZ": "PZ",
    "PO4": "O2",
    "PO8": "O2",

    "OZ": "O1",
}



CHISCO_MAPPING = {
    # direct 10-20 matches
    "FP1": "FP1", "FP2": "FP2",
    "F7": "F7", "F3": "F3", "FZ": "FZ", "F4": "F4", "F8": "F8",
    "T7": "T7", "C3": "C3", "CZ": "CZ", "C4": "C4", "T8": "T8",
    "P7": "P7", "P3": "P3", "PZ": "PZ", "P4": "P4", "P8": "P8",
    "O1": "O1", "O2": "O2",

    # midline extended
    "FPZ":  "FZ",
    "AFZ":  "FZ",
    "FCZ":  "FZ",
    "FCCZ": "CZ",
    "CPPZ": "PZ",
    "POZ":  "PZ",
    "PPOZ": "PZ",
    "POOZ": "O1",
    "OZ":   "O1",

    # frontal extended
    "AF3": "F3", "AF4": "F4",
    "AF7": "F7", "AF8": "F8",
    "F1": "FZ",  "F2": "FZ",
    "F5": "F7",  "F6": "F8",
    "FC1": "F3", "FC2": "F4",
    "FC3": "F3", "FC4": "F4",
    "FC5": "F7", "FC6": "F8",

    # fronto-temporal
    "FT7": "T7", "FT8": "T8",
    "FT9": "T7", "FT10": "T8",

    # half-step frontal (extended 10-20 'h' electrodes)
    "FFC1H": "FZ",  "FFC2H": "FZ",
    "FFC3H": "F3",  "FFC4H": "F4",
    "FFC5H": "F7",  "FFC6H": "F8",
    "FFT7H": "F7",  "FFT8H": "F8",
    "AFF5H": "F7",  "AFF6H": "F8",

    # fronto-central (between FC and C)
    "FCC1H": "CZ",  "FCC2H": "CZ",
    "FCC3H": "C3",  "FCC4H": "C4",
    "FCC5H": "C3",  "FCC6H": "C4",

    # fronto-temporal half-step
    "FTT7H": "T7",   "FTT8H": "T8",
    "FTT9H": "T7",   "FTT10H": "T8",

    # central extended
    "C1": "CZ", "C2": "CZ",
    "C5": "C3", "C6": "C4",

    # centro-parietal half-step
    "CCP1H": "PZ",  "CCP2H": "PZ",
    "CCP3H": "P3",  "CCP4H": "P4",
    "CCP5H": "P7",  "CCP6H": "P8",

    # temporal-parietal
    "T9": "T7",   "T10": "T8",
    "TP7": "T7",  "TP8": "T8",
    "TTP7H": "T7", "TTP8H": "T8",
    "TPP5H": "P7", "TPP8H": "P8",

    # parietal extended
    "P5": "P7",  "P6": "P8",
    "P9": "P7",  "P10": "P8",
    "P11": "P7", "P12": "P8",
    "CP1": "PZ", "CP2": "PZ",
    "CP3": "P3", "CP4": "P4",

    # centro-parietal (half-step, parietal side)
    "CPP1H": "PZ",  "CPP2H": "PZ",
    "CPP3H": "P3",  "CPP4H": "P4",
    "CPP5H": "P7",  "CPP6H": "P8",

    # parieto-occipital
    "PO1": "O1",  "PO2": "O2",
    "PO3": "O1",  "PO4": "O2",
    "PO9": "O1",  "PO10": "O2",
    "PO11": "O1", "PO12": "O2",
    "PPO1": "O1", "PPO2": "O2",
    "PPO7": "O1", "PPO8": "O2",

    # occipital half-step
    "OI1": "O1",    "OI2": "O2",
    "POO3": "O1",   "POO4": "O2",
    "POO7": "O1",   "POO8": "O2",
    "POO9H": "O1",  "POO10H": "O2",
    "POO11H": "O1", "POO12H": "O2",
}

BCI2020_MAPPING = {
    "FP1": "FP1",
    "FP2": "FP2",

    "F3": "F3",
    "F4": "F4",
    "F7": "F7",
    "F8": "F8",
    "FZ": "FZ",

    "C3": "C3",
    "C4": "C4",
    "CZ": "CZ",

    "P3": "P3",
    "P4": "P4",
    "P7": "P7",
    "P8": "P8",
    "PZ": "PZ",

    "O1": "O1",
    "O2": "O2",

    "T7": "T7",
    "T8": "T8",

    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",

    "FC5": "F7",
    "FC3": "F3",
    "FC1": "F3",
    "FCZ": "FZ",
    "FC2": "F4",
    "FC4": "F4",
    "FC6": "F8",

    "CP5": "P7",
    "CP3": "P3",
    "CP1": "P3",
    "CPZ": "PZ",
    "CP2": "P4",
    "CP4": "P4",
    "CP6": "P8",

    "PO7": "O1",
    "PO3": "O1",
    "POZ": "PZ",
    "PO4": "O2",
    "PO8": "O2",

    "OZ": "O1",
}



def _normalize_ch_name(ch_name: str) -> str:
    if ch_name is None:
        return ""
    s = str(ch_name).strip()
    s = _PREFIX_PAT.sub("", s)
    s = _SUFFIX_PAT.sub("", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.upper()
    s = _JUNK_PAT.sub("", s)
    s = _KEEP_PAT.sub("", s)
    if s.endswith(".."):
        s = s[:-2]
    # print(f"Normalized channel name: '{ch_name}' -> '{s}'")
    return s

def normalize_channels(ch_names: List[str]) -> List[str]:
    return [_normalize_ch_name(ch) for ch in ch_names]

def is_eeg_channel(ch_name: str) -> bool:
    raw = str(ch_name) if ch_name is not None else ""
    if _NON_EEG_PAT.search(raw):
        return False
    n = _normalize_ch_name(raw)
    if not n:
        return False

    if n.isdigit():
        return True

    if "-" in n:
        a, b = n.split("-", 1)
        return (a in EEG_LIKE_SET) and (b.split("-", 1)[0] in EEG_LIKE_SET)

    return n in EEG_LIKE_SET

def eeg_channels(ch_names: List[str]) -> List[str]:
    return [ch for ch in ch_names if is_eeg_channel(ch)]


def reorder_and_pad(x: torch.Tensor, kept_names: list[str], target_chs: list[str] = TARGET_CHS):
    device = x.device
    dtype = x.dtype
    T = x.shape[1]

    x_out = torch.zeros((len(target_chs), T), dtype=dtype, device=device)
    mask  = torch.zeros((len(target_chs),), dtype=torch.bool, device=device)

    target_pos = {ch: i for i, ch in enumerate(target_chs)}

    for src_i, ch in enumerate(kept_names):
        j = target_pos.get(ch, None)
        if j is None:
            continue
        x_out[j] = x[src_i]
        mask[j] = True

    return x_out, mask

