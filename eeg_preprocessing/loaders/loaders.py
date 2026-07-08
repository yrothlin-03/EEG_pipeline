import random
import lmdb
import pickle
import torch
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple
from torch.utils.data import Dataset, DataLoader
from .dataset import CustomDataset
from logging import Logger
import numpy as np
import matplotlib.pyplot as plt
from .channel_mapping import TARGET_CHS




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


def split_subjects(all_subjects, split_ratio, seed=42):
    random.seed(seed)
    subjects = all_subjects.copy()
    random.shuffle(subjects)
    n_total = len(subjects)

    if n_total == 0:
        return [], [], []

    r_train, r_val, r_test = split_ratio

    n_train = round(r_train * n_total)
    n_val   = round(r_val * n_total)
    n_test  = round(r_test * n_total)

    total = n_train + n_val + n_test

    if total != n_total:
        n_train += n_total - total

    if n_total >= 3:
        if r_train > 0 and n_train == 0:
            n_train = 1
        if r_val > 0 and n_val == 0:
            n_val = 1
        if r_test > 0 and n_test == 0:
            n_test = 1

    total = n_train + n_val + n_test
    if total > n_total:
        overflow = total - n_total
        for name in ("train", "val", "test"):
            if overflow == 0:
                break
            if name == "train" and n_train > 1:
                n_train -= 1
            elif name == "val" and n_val > 1:
                n_val -= 1
            elif name == "test" and n_test > 1:
                n_test -= 1
            overflow = n_train + n_val + n_test - n_total

    train = subjects[:n_train]
    val   = subjects[n_train:n_train + n_val]
    test  = subjects[n_train + n_val:n_train + n_val + n_test]

    return train, val, test


def _to_id_list(ids: Sequence[str] | None) -> list[str] | None:
    if ids is None:
        return None
    return [str(sid) for sid in ids]


def _unpack_split_ids(
    split_ids: Mapping[str, Sequence[str] | None] | Sequence[Sequence[str] | None] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    if split_ids is None:
        return None, None, None

    if isinstance(split_ids, Mapping):
        return (
            _to_id_list(split_ids.get("train")),
            _to_id_list(split_ids.get("val", split_ids.get("validation"))),
            _to_id_list(split_ids.get("test")),
        )

    if len(split_ids) != 3:
        raise ValueError("split_ids must contain exactly three lists: train, val, test.")

    train_ids, val_ids, test_ids = split_ids
    return _to_id_list(train_ids), _to_id_list(val_ids), _to_id_list(test_ids)


def resolve_subject_splits(
    all_subjects: Sequence[str],
    split_ratio: Tuple[float, float, float],
    seed: int = 42,
    split_ids: Mapping[str, Sequence[str] | None] | Sequence[Sequence[str] | None] | None = None,
    train_ids: Sequence[str] | None = None,
    val_ids: Sequence[str] | None = None,
    test_ids: Sequence[str] | None = None,
):
    explicit_train, explicit_val, explicit_test = _unpack_split_ids(split_ids)

    if train_ids is not None:
        explicit_train = _to_id_list(train_ids)
    if val_ids is not None:
        explicit_val = _to_id_list(val_ids)
    if test_ids is not None:
        explicit_test = _to_id_list(test_ids)

    explicit = {
        "train": explicit_train,
        "val": explicit_val,
        "test": explicit_test,
    }

    if all(ids is None for ids in explicit.values()):
        return split_subjects(list(all_subjects), split_ratio=split_ratio, seed=seed), "ratio"

    known_subjects = set(all_subjects)
    seen_by_split = {}
    for split_name, ids in explicit.items():
        if ids is None:
            continue

        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicates:
            raise ValueError(f"Duplicate subject IDs in {split_name} split: {duplicates}")

        unknown = sorted(set(ids) - known_subjects)
        if unknown:
            raise ValueError(f"Unknown subject IDs in {split_name} split: {unknown}")

        for sid in ids:
            if sid in seen_by_split:
                raise ValueError(
                    f"Subject ID {sid!r} is present in both "
                    f"{seen_by_split[sid]!r} and {split_name!r} splits."
                )
            seen_by_split[sid] = split_name

    remaining_subjects = [sid for sid in all_subjects if sid not in seen_by_split]
    missing_splits = [name for name, ids in explicit.items() if ids is None]

    generated = {"train": [], "val": [], "test": []}
    if missing_splits and remaining_subjects:
        ratio_by_split = {
            "train": float(split_ratio[0]),
            "val": float(split_ratio[1]),
            "test": float(split_ratio[2]),
        }
        missing_ratio_sum = sum(ratio_by_split[name] for name in missing_splits)
        if missing_ratio_sum > 0:
            adjusted_ratio = tuple(
                ratio_by_split[name] / missing_ratio_sum if name in missing_splits else 0.0
                for name in ("train", "val", "test")
            )
            gen_train, gen_val, gen_test = split_subjects(
                remaining_subjects, split_ratio=adjusted_ratio, seed=seed
            )
            generated.update({"train": gen_train, "val": gen_val, "test": gen_test})

    train_subjects = explicit_train if explicit_train is not None else generated["train"]
    val_subjects = explicit_val if explicit_val is not None else generated["val"]
    test_subjects = explicit_test if explicit_test is not None else generated["test"]

    return (train_subjects, val_subjects, test_subjects), "ids"




def collate(batch):
    if len(batch[0]) == 3:
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        ch_names = [b[2] for b in batch]
        return xs, ys, ch_names

    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys



def build_loaders(
    lmdb_path: str | Path,
    split_ratio: Tuple[float, float, float],
    batch_size: int,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    shuffle_val:  bool = False,
    logger: Logger = None,
    channel_mode: str = "mapped",
    return_ch_names: bool = False,
    split_ids: Mapping[str, Sequence[str] | None] | Sequence[Sequence[str] | None] | None = None,
    train_ids: Sequence[str] | None = None,
    val_ids: Sequence[str] | None = None,
    test_ids: Sequence[str] | None = None,
):
    lmdb_path = str(lmdb_path)

    all_subjects = get_subject_ids(Path(lmdb_path))
    (train_subjects, val_subjects, test_subjects), split_source = resolve_subject_splits(
        all_subjects,
        split_ratio=split_ratio,
        seed=seed,
        split_ids=split_ids,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    if logger:
        logger.info(f"[LOADER] Total subjects: {len(all_subjects)}")
        logger.info(f"[LOADER] Split source: {split_source}")
        logger.info(f"[LOADER] Train subjects: {len(train_subjects)} | {train_subjects}")
        logger.info(f"[LOADER] Validation subjects: {len(val_subjects)} | {val_subjects}")
        logger.info(f"[LOADER] Test subjects: {len(test_subjects)} | {test_subjects} ")

    print(f"[LOADER] Total subjects: {len(all_subjects)}")
    print(f"[LOADER] Split source: {split_source}")
    print(f"[LOADER] Train subjects: {len(train_subjects)} | {train_subjects}")
    print(f"[LOADER] Validation subjects: {len(val_subjects)} | {val_subjects}")
    print(f"[LOADER] Test subjects: {len(test_subjects)} | {test_subjects}")

    train_dataset = CustomDataset(lmdb_path, train_subjects, channel_mode=channel_mode, return_ch_names=return_ch_names)
    val_dataset = CustomDataset(lmdb_path, val_subjects, channel_mode=channel_mode, return_ch_names=return_ch_names)
    test_dataset = CustomDataset(lmdb_path, test_subjects, channel_mode=channel_mode, return_ch_names=return_ch_names)
    shuffle_val = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        collate_fn=collate,
    )

    test_loader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            collate_fn=collate,
        )

    return train_loader, val_loader, test_loader




def plot_random_eeg_samples(
    ds: Dataset,
    out_dir: str | Path,
    n_samples: int = 8,
    n_channels: int = 6,
    seed: int = 42,
    t_start: int | None = None,
    t_len: int | None = None,
    channel_names: Sequence[str] = TARGET_CHS,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    for k, idx in enumerate(idxs):
        x, y = ds[int(idx)]
        x_np = x.detach().cpu().numpy()
        C, T = x_np.shape

        if t_len is not None:
            if t_start is None:
                s = int(rng.integers(0, max(1, T - t_len)))
            else:
                s = int(max(0, min(t_start, T - 1)))
            e = int(min(T, s + t_len))
        else:
            s, e = 0, T

        ch_k = min(n_channels, C)
        ch_idx = rng.choice(C, size=ch_k, replace=False)

        plt.figure(figsize=(14, 2.2 * ch_k))
        for j, ch in enumerate(ch_idx):
            ax = plt.subplot(ch_k, 1, j + 1)
            ax.plot(x_np[ch, s:e])
            name = channel_names[ch] if ch < len(channel_names) else f"Ch{ch}"
            ax.set_title(f"idx={int(idx)} y={int(y)} {name}")
            ax.set_xlim(0, (e - s) - 1)

        plt.tight_layout()
        save_path = out_dir / f"eeg_idx{int(idx)}_y{int(y)}_k{k}.png"
        print(f"Saving EEG sample plot to {save_path.as_posix()}")
        plt.savefig(save_path)
        plt.close()



