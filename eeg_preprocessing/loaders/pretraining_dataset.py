from pathlib import Path
from typing import List, Literal, Sequence, Tuple, Mapping

import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .channel_mapping import (
    _normalize_ch_name,
    reorder_and_pad,
    is_eeg_channel,
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
from .loaders import get_subject_ids, resolve_subject_splits


ChannelMode = Literal["raw", "mapped"]

_NAME_TO_MAPPING = {
    "tueg": TUEG_MAPPING,
    "tuab": TUAB_MAPPING,
    "physionetmi": PHYSIONET_MAPPING,
    "sleepedf": SLEEPEDF_MAPPING,
    "seedv": SEEDV_MAPPING,
    "bci2a": BCI2A_MAPPING,
    "faced": FACED_MAPPING,
    "siena": SIENA_MAPPING,
    "shumi": SHUMI_MAPPING,
    "chbmit": CHBMIT_MAPPING,
    "bci2020": BCI2020_MAPPING,
    "karaone": KARAONE_MAPPING,
    "chisco": CHISCO_MAPPING,
}


def _mapping_from_path(lmdb_path: str):
    name = Path(lmdb_path).stem.lower()
    for prefix, mapping in _NAME_TO_MAPPING.items():
        if name.startswith(prefix):
            return mapping
    raise ValueError(f"Unknown dataset name from path stem: {Path(lmdb_path).stem}")


class EEGAugment:
    def __init__(
        self,
        time_shift_ratio: float = 0.1,
        amplitude_range: Tuple[float, float] = (0.8, 1.2),
        max_channel_dropout: int = 2,
        noise_std: float = 0.01,
        time_mask_ratio: Tuple[float, float] = (0.05, 0.10),
        p: float = 0.5,
        generator: torch.Generator | None = None,
    ):
        self.time_shift_ratio = time_shift_ratio
        self.amplitude_range = amplitude_range
        self.max_channel_dropout = max_channel_dropout
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio
        self.p = p
        self.generator = generator

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.generator).item())

    def _randint(self, low: int, high: int) -> int:
        # inclusive low, exclusive high
        if high <= low:
            return low
        return int(torch.randint(low, high, (1,), generator=self.generator).item())

    def _apply_views(self, x: torch.Tensor) -> torch.Tensor:
        """Label-preserving transforms (applied to both input and target)."""
        T = x.shape[1]

        # --- Time shift +/- time_shift_ratio (circular roll) ---
        if self.time_shift_ratio > 0 and self._rand() < self.p:
            max_shift = int(self.time_shift_ratio * T)
            if max_shift > 0:
                shift = self._randint(-max_shift, max_shift + 1)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=1)

        # --- Amplitude scaling x U(lo, hi) ---
        if self._rand() < self.p:
            lo, hi = self.amplitude_range
            scale = lo + (hi - lo) * self._rand()
            x = x * scale

        return x

    def _apply_corruptions(self, x: torch.Tensor) -> torch.Tensor:
        """Destructive transforms (applied to the input only)."""
        C, T = x.shape

        # --- Channel dropout (1..max channels zeroed) ---
        if self.max_channel_dropout > 0 and self._rand() < self.p:
            n_drop = self._randint(1, min(self.max_channel_dropout, C) + 1)
            perm = torch.randperm(C, generator=self.generator)[:n_drop]
            x[perm] = 0.0

        # --- Additive Gaussian noise ---
        if self.noise_std > 0 and self._rand() < self.p:
            noise = torch.randn(x.shape, generator=self.generator) * self.noise_std
            x = x + noise

        # --- Time masking (contiguous block zeroed) ---
        if self._rand() < self.p:
            lo, hi = self.time_mask_ratio
            mask_ratio = lo + (hi - lo) * self._rand()
            mask_len = int(mask_ratio * T)
            if mask_len > 0:
                start = self._randint(0, T - mask_len + 1)
                x[:, start:start + mask_len] = 0.0

        return x

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input, target)``: corrupted input + uncorrupted view target."""
        if x.ndim != 2:
            raise ValueError(f"EEGAugment expects a (C, T) tensor, got shape {tuple(x.shape)}")

        # Views shared by input and target keep the target consistent yet uncorrupted.
        target = self._apply_views(x.clone())
        # Corruptions live on the input only: corrupted_input -> view_target.
        inp = self._apply_corruptions(target.clone())
        return inp, target


class PretrainingDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        subject_ids: Sequence[str] | None = None,
        channel_mode: ChannelMode = "mapped",
        augment: bool = False,
        augmenter: EEGAugment | None = None,
        return_ch_names: bool = False,
        ids: Sequence[str] | None = None,
        scale_factor: float = 1.0,
    ):
        self.lmdb_path = str(lmdb_path)

        if subject_ids is not None and ids is not None and list(subject_ids) != list(ids):
            raise ValueError("Provide either subject_ids or ids, not two different lists.")

        selected_ids = ids if ids is not None else subject_ids
        if selected_ids is None:
            raise ValueError("PretrainingDataset requires a list of subject IDs via subject_ids or ids.")

        self.subject_ids = list(selected_ids)
        self.channel_mode = channel_mode
        self.augment = augment
        self.scale_factor = scale_factor
        self.augmenter = augmenter if augmenter is not None else EEGAugment()
        self.return_ch_names = return_ch_names
        self._cache = {}
        self.env = None

        self.mapping = _mapping_from_path(self.lmdb_path)
        self.keep_set = set(self.mapping.values())

        env = lmdb.open(
            self.lmdb_path,
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

        def prio(src_name_norm: str) -> int:
            return 0 if src_name_norm in {"FZ", "C3", "CZ", "C4", "PZ"} else 1

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

    def __getitem__(self, i):
        self._init_env()

        global_idx = int(self.indices[i])
        key = global_idx.to_bytes(8, "big")

        with self.env.begin(write=False) as txn:
            blob = txn.get(key)

        rec = pickle.loads(blob)
        x = torch.from_numpy(rec["x"]).float()
        if self.scale_factor != 1.0:
            x = x * self.scale_factor
        ch_names = rec.get("ch_names", [])

        if self.channel_mode == "raw":
            eeg_idx = [j for j, ch in enumerate(ch_names) if is_eeg_channel(ch)]
            if len(eeg_idx) > 0:
                x = x[eeg_idx].contiguous()
                kept_ch_names = [ch_names[j] for j in eeg_idx]
            else:
                kept_ch_names = ch_names
        else:
            keep_idx, keep_names = self._keep(ch_names)
            x = x[keep_idx].contiguous()
            x, kept_ch_names = reorder_and_pad(x, keep_names, TARGET_CHS)

        if self.augment:
            x_in, x_tgt = self.augmenter(x)
        else:
            x_in = x_tgt = x

        if self.return_ch_names:
            return x_in, x_tgt, kept_ch_names
        return x_in, x_tgt


# Constructor hyperparameters accepted from an ``augment`` config block. The
# ``enabled`` flag is handled separately via ``augment_train`` and ignored here.
_AUGMENT_KEYS = (
    "time_shift_ratio",
    "amplitude_range",
    "max_channel_dropout",
    "noise_std",
    "time_mask_ratio",
    "p",
)


def _build_augmenter(augment_config: Mapping[str, object] | None) -> EEGAugment:
    kwargs = {}
    if augment_config:
        kwargs = {k: augment_config[k] for k in _AUGMENT_KEYS if k in augment_config}
    return EEGAugment(**kwargs)


def _pretraining_collate(batch):
    x_in = torch.stack([b[0] for b in batch], dim=0)
    x_tgt = torch.stack([b[1] for b in batch], dim=0)
    if len(batch[0]) == 3:
        ch_names = [b[2] for b in batch]
        return x_in, x_tgt, ch_names
    return x_in, x_tgt


def get_pretraining_loader(
    lmdb_paths: str | Path | Sequence[str | Path],
    split_ratio: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    channel_mode: ChannelMode = "mapped",
    return_ch_names: bool = False,
    augment_train: bool = True,
    augment_config: Mapping[str, object] | None = None,
    scale_factor: float = 1e6,
    drop_last: bool = True,
    split_ids: Mapping[str, Sequence[str] | None] | Sequence[Sequence[str] | None] | None = None,
):
    if isinstance(lmdb_paths, (str, Path)):
        lmdb_paths = [lmdb_paths]
    lmdb_paths = [str(p) for p in lmdb_paths]

    augmenter = _build_augmenter(augment_config) if augment_train else None

    train_sets, val_sets = [], []
    for lmdb_path in lmdb_paths:
        all_subjects = get_subject_ids(Path(lmdb_path))
        (train_subjects, val_subjects, _), _ = resolve_subject_splits(
            all_subjects,
            split_ratio=split_ratio,
            seed=seed,
            split_ids=split_ids,
        )

        if train_subjects:
            train_sets.append(
                PretrainingDataset(
                    lmdb_path,
                    subject_ids=train_subjects,
                    channel_mode=channel_mode,
                    augment=augment_train,
                    augmenter=augmenter,
                    return_ch_names=return_ch_names,
                    scale_factor=scale_factor,
                )
            )
        if val_subjects:
            val_sets.append(
                PretrainingDataset(
                    lmdb_path,
                    subject_ids=val_subjects,
                    channel_mode=channel_mode,
                    augment=False,
                    return_ch_names=return_ch_names,
                    scale_factor=scale_factor,
                )
            )

    if not train_sets:
        raise ValueError("No training subjects resolved from the provided LMDB store(s).")

    train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_dataset = None
    if val_sets:
        val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        collate_fn=_pretraining_collate,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            collate_fn=_pretraining_collate,
        )

    return train_loader, val_loader
