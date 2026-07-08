from pathlib import Path
from typing import List, Literal, Sequence
import lmdb
import pickle
import re
import torch
from torch.utils.data import Dataset
from .channel_mapping import (
    _normalize_ch_name,
    reorder_and_pad,
    is_eeg_channel,
    EEG_LIKE_SET,
    _NON_EEG_PAT,
    TARGET_CHS,
    TUEG_MAPPING,
    TUAB_MAPPING,
    PHYSIONET_MAPPING,
    SLEEPEDF_MAPPING,
    SEEDV_MAPPING,
    BCI2A_MAPPING,
    FACED_MAPPING,
    SIENA_MAPPING,
    SHUMI_MAPPING,
    CHBMIT_MAPPING,
    BCI2020_MAPPING,
    KARAONE_MAPPING,
    CHISCO_MAPPING,

)



def get_subject_ids(lmdb_path: Path) -> List[str]:
    env = lmdb.open(
        lmdb_path.as_posix(),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=False) as txn:
            subject_ids = pickle.loads(txn.get(b"__subject_ids__"))
        return list(subject_ids)
    finally:
        env.close()

ChannelMode = Literal["raw", "mapped"]


class CustomDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        subject_ids: Sequence[str] | None = None,
        channel_mode: ChannelMode = "mapped",
        debug=False,
        return_ch_names: bool = False,
        ids: Sequence[str] | None = None,
    ):
        self.lmdb_path = lmdb_path
        if subject_ids is not None and ids is not None and list(subject_ids) != list(ids):
            raise ValueError("Provide either subject_ids or ids, not two different lists.")

        selected_ids = ids if ids is not None else subject_ids
        if selected_ids is None:
            raise ValueError("CustomDataset requires a list of subject IDs via subject_ids or ids.")

        self.subject_ids = list(selected_ids)
        self.channel_mode = channel_mode
        self.debug = debug
        self.return_ch_names = return_ch_names
        self._cache = {}
        self.env = None

        p = Path(lmdb_path)
        name = p.stem.lower()
        if name.startswith("tueg"):
            self.mapping = TUEG_MAPPING
        elif name.startswith("tuab"):
            self.mapping = TUAB_MAPPING
        elif name.startswith("physionetmi"):
            self.mapping = PHYSIONET_MAPPING
        elif name.startswith("sleepedf"):
            self.mapping = SLEEPEDF_MAPPING
        elif name.startswith("seedv"):
            self.mapping = SEEDV_MAPPING
        elif name.startswith("bci2a"):
            self.mapping = BCI2A_MAPPING
        elif name.startswith("faced"):
            self.mapping = FACED_MAPPING
        elif name.startswith("siena"):
            self.mapping = SIENA_MAPPING
        elif name.startswith("shumi"):
            self.mapping = SHUMI_MAPPING
        elif name.startswith("chbmit"):
            self.mapping = CHBMIT_MAPPING
        elif name.startswith("bci2020"):
            self.mapping = BCI2020_MAPPING
        elif name.startswith("karaone"):
            self.mapping = KARAONE_MAPPING
        elif name.startswith("chisco"):
            self.mapping = CHISCO_MAPPING
        else:
            raise ValueError(f"Unknown dataset name from path stem: {p.stem}")

        self.keep_set = set(self.mapping.values())

        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_dbs=1,
        )
        with env.begin(write=False) as txn:
            self.sub2range = pickle.loads(txn.get(b"__subj_ranges__"))
            self.total_len = pickle.loads(txn.get(b"__len__"))
        env.close()

        self.indices = []
        for sid in self.subject_ids:
            for start, end in self.sub2range.get(sid, []):
                if end > start:
                    self.indices.extend(range(start, end))
        self.indices = [i for i in self.indices if 0 <= i < self.total_len]

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_dbs=1,
            )

    def __len__(self):
        return len(self.indices)

    def _keep(self, ch_names):
        key = tuple(ch_names)
        if key in self._cache:
            return self._cache[key]

        target_pos = {ch: i for i, ch in enumerate(TARGET_CHS)}

        def prio(src_name_norm: str) -> int:
            return 0 if src_name_norm in {"FZ","C3","CZ","C4","PZ"} else 1

        best = {} 
        for i, ch in enumerate(ch_names):
            n = _normalize_ch_name(ch)
            m = self.mapping.get(n)
            if m is None or m not in self.keep_set:
                continue

            p = prio(n)
            if (m not in best) or (p < best[m][0]):
                best[m] = (p, i)

        keep_idx, keep_names = [], []
        for m in TARGET_CHS:
            if m in best:
                keep_idx.append(best[m][1])
                keep_names.append(m)

        self._cache[key] = (keep_idx, keep_names)
        return self._cache[key]

    def _get_channels_info(self, idx: int = 0):
        self._init_env()

        global_idx = int(self.indices[idx])
        key = global_idx.to_bytes(8, "big")

        with self.env.begin(write=False) as txn:
            blob = txn.get(key)

        rec = pickle.loads(blob)
        ch_names = rec.get("ch_names", [])

        mapping_used = {}
        kept_names = []

        if self.channel_mode == "raw":
            eeg_idx = [i for i, ch in enumerate(ch_names) if is_eeg_channel(ch)]
            kept_names = [ch_names[i] for i in eeg_idx]

            for ch in kept_names:
                mapping_used[ch] = ch

        else:
            keep_idx, kept_names = self._keep(ch_names)

            for i, ch in enumerate(ch_names):
                n = _normalize_ch_name(ch)
                m = self.mapping.get(n)
                if m in kept_names:
                    mapping_used[ch] = m

        return {
            "original_channels": ch_names,
            "kept_channels": kept_names,
            "mapping": mapping_used,
            "n_original_channels": len(ch_names),
            "n_kept_channels": len(kept_names),
        }
        
    def __getitem__(self, i):
        self._init_env()

        global_idx = int(self.indices[i])
        key = global_idx.to_bytes(8, "big")

        with self.env.begin(write=False) as txn:
            blob = txn.get(key)

        rec = pickle.loads(blob)

        x = torch.from_numpy(rec["x"])
        y = int(rec["label"])
        ch_names = rec.get("ch_names", [])

        if self.channel_mode == "raw":
            eeg_idx = [j for j, ch in enumerate(ch_names) if is_eeg_channel(ch)]
            kept_ch_names = [ch_names[j] for j in eeg_idx]

            if len(eeg_idx) > 0:
                x = x[eeg_idx].contiguous()
            else:
                kept_ch_names = ch_names

            if self.return_ch_names:
                return x, y, kept_ch_names
            return x, y

        keep_idx, keep_names = self._keep(ch_names)
        x = x[keep_idx].contiguous()
        x, kept_ch_names = reorder_and_pad(x, keep_names, TARGET_CHS)

        if self.return_ch_names:
            return x, y, kept_ch_names
        return x, y
 


def debug_idx(ds: CustomDataset, i: int):
    global_idx = int(ds.indices[i])
    key = global_idx.to_bytes(8, "big")
    with ds.env.begin(write=False) as txn:
        blob = txn.get(key)
    rec = pickle.loads(blob)

    orig = rec["ch_names"]
    keep_idx, keep_names = ds._keep(orig)

    print("dataset idx:", i, "global_idx:", global_idx)
    print("orig n_ch:", len(orig))
    print("kept n_ch:", len(keep_names))
    print("kept:", keep_names)

    missing = [c for c in TARGET_CHS if c not in keep_names]
    print("missing from TARGET_CHS:", missing)

