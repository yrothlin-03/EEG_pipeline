from pathlib import Path
from typing import List
import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from .channel_mapping import (
    TARGET_CHS,
    TUEG_MAPPING,
    TUAB_MAPPING,
    PHYSIONET_MAPPING,
    SLEEPEDFX_MAPPING,
    SEEDV_MAPPING,
    BCI2A_MAPPING,
    FACED_MAPPING,
)



def _normalize_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    s = s.replace(".", "")
    s = s.replace("EEG ", "")
    s = s.replace("EEG-", "")
    s = s.split("-")[0].strip()
    return s




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



class CustomDataset(Dataset):
    def __init__(self, lmdb_path: str, subject_ids: List[str]):
        self.lmdb_path = lmdb_path
        self.subject_ids = list(subject_ids)
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
        elif name.startswith("sleepedfx"):
            self.mapping = SLEEPEDFX_MAPPING
        elif name.startswith("seedv"):
            self.mapping = SEEDV_MAPPING
        elif name.startswith("bci2a"):
            self.mapping = BCI2A_MAPPING
        elif name.startswith("faced"):
            self.mapping = FACED_MAPPING
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
            self.sub2range = pickle.loads(txn.get(b"__subject_ranges__"))
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

        # priorité: canaux explicitement nommés (FZ, C3, CZ, C4, PZ) > canaux indexés (0..16)
        def prio(src_name_norm: str) -> int:
            return 0 if src_name_norm in {"FZ","C3","CZ","C4","PZ"} else 1

        best = {}  # mapped_name -> (priority, src_i)
        for i, ch in enumerate(ch_names):
            n = _normalize_ch_name(ch)
            m = self.mapping.get(n)
            if m is None or m not in self.keep_set:
                continue

            p = prio(n)
            if (m not in best) or (p < best[m][0]):
                best[m] = (p, i)

        # construire keep_idx/keep_names dans l'ordre des TARGET_CHS pour être stable
        keep_idx, keep_names = [], []
        for m in TARGET_CHS:
            if m in best:
                keep_idx.append(best[m][1])
                keep_names.append(m)

        self._cache[key] = (keep_idx, keep_names)
        return self._cache[key]

    def __getitem__(self, i):
        self._init_env()

        global_idx = int(self.indices[i])
        key = global_idx.to_bytes(8, "big")

        with self.env.begin(write=False) as txn:
            blob = txn.get(key)
        rec = pickle.loads(blob)

        x = torch.from_numpy(rec["x"])
        # print("Original channels:", rec["ch_names"])
        keep_idx, keep_names = self._keep(rec["ch_names"])
        # print("Kept channels:", keep_names)
        x = x[keep_idx].contiguous()
        x, _ = reorder_and_pad(x, keep_names, TARGET_CHS)
        y = int(rec["label"])
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


if __name__ == "__main__":
    pass