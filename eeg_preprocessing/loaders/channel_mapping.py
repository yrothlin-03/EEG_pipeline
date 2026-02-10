import re


EEG_LIKE_SET = {
    "FP1","FP2","FPZ","AFZ",
    "F7","F3","F1","FZ","F2","F4","F8",
    "FC5","FC3","FC1","FCZ","FC2","FC4","FC6",
    "T7","T3","C5","C3","C1","CZ","C2","C4","C6","T4","T8",
    "CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8",
    "O1","OZ","O2",
}

_NON_EEG_PAT = re.compile(
    r"(EOG|EMG|ECG|EKG|PPG|PLETH|RESP|RESPIR|AIRFLOW|NASAL|THOR|ABDO|"
    r"SNORE|MIC|AUX|TRIG|TRIGGER|EVENT|MARK|REF|GND|GROUND|A1|A2|"
    r"M1|M2|EAR|MASTOID|CHIN|LEG|ARM|CHEST|BELT|TEMP|HEART|PULSE|"
    r"SaO2|SPO2|CO2|ETCO2|GSR|EDA|ACC|GYRO|MAG|PHOTO|PHOTIC)",
    re.IGNORECASE
)



TARGET_CHS = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","FZ","CZ","PZ"]


TUEG_MAPPING = {
    "FP1": "FP1",
    "FP2": "FP2",
    "F7": "F7",
    "F3": "F3",
    "FZ": "FZ",
    "F4": "F4",
    "F8": "F8",

    "T3": "T7",
    "C3": "C3",
    "CZ": "CZ",
    "C4": "C4",
    "T4": "T8",

    "T5": "P7",
    "P3": "P3",
    "PZ": "PZ",
    "P4": "P4",
    "T6": "P8",

    "O1": "O1",
    "O2": "O2",
}


TUAB_MAPPING = TUEG_MAPPING.copy()


PHYSIONET_MAPPING = {
    "FP1":"FP1","FP2":"FP2","F7":"F7","F3":"F3","FZ":"FZ","F4":"F4","F8":"F8",
    "T7":"T7","T8":"T8","C3":"C3","CZ":"CZ","C4":"C4",
    "P7":"P7","P3":"P3","PZ":"PZ","P4":"P4","P8":"P8",
    "O1":"O1","O2":"O2",
}

SLEEPEDFX_MAPPING = {
    "FPZ": "FZ",   
    "CZ":  "CZ",   
    "PZ":  "PZ",   
    "OZ":  "PZ",   
}


SEEDV_MAPPING = {
    "FP1": "FP1",
    "FP2": "FP2",
    "F7":  "F7",
    "F3":  "F3",
    "FZ":  "FZ",
    "F4":  "F4",
    "F8":  "F8",
    "T7":  "T7",
    "T8":  "T8",
    "C3":  "C3",
    "CZ":  "CZ",
    "C4":  "C4",
    "P7":  "P7",
    "P3":  "P3",
    "PZ":  "PZ",
    "P4":  "P4",
    "P8":  "P8",
    "O1":  "O1",
    "O2":  "O2",

    "FPZ": "FZ",
    "FCZ": "FZ",
    "CPZ": "PZ",
    "POZ": "PZ",

    "AF3": "F3",
    "AF4": "F4",
    "F1":  "FZ",
    "F2":  "FZ",
    "FC1": "F3",
    "FC2": "F4",
    "C1":  "CZ",
    "C2":  "CZ",
    "CP1": "P3",
    "CP2": "P4",
    "P1":  "PZ",
    "P2":  "PZ",
    "PO3": "O1",
    "PO4": "O2",
    "PO7": "O1",
    "PO8": "O2",
    "OZ":  "PZ",  # bof bof, à redéfinir mais flemme là tout de suite

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


FACED_MAPPING = {
    ch: ch for ch in FACED_CHANNEL_LIST
}



BCI2A_MAPPING = {
    "FZ": "FZ",

    "0": "F3",   
    "1": "F3",   
    "2": "FZ",   
    "3": "FZ",   
    "4": "F4",   
    "5": "F4",   

    "C3": "C3",
    "6": "CZ",   
    "CZ": "CZ",
    "7": "CZ",   
    "C4": "C4",
    "8": "CZ",   

    "9":  "P3",  
    "10": "PZ",  
    "11": "PZ",  
    "12": "PZ",  
    "13": "P4",  
    "14": "P3",  
    "PZ": "PZ",
    "15": "P4",  
    "16": "PZ",  
}



SIENA_MAPPING = {}

SHUMI_MAPPING = {}

CHBMIT_MAPPING = {}