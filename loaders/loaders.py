import random
import lmdb
import pickle
import torch
from pathlib import Path
from typing import List, Tuple, Sequence
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

    n_train = int(split_ratio[0] * n_total)
    n_val   = int(split_ratio[1] * n_total)
    n_test  = int(split_ratio[2] * n_total)

    if n_total >= 3 and n_val == 0 and split_ratio[1] > 0:
        n_val = 1
    if n_total >= 2 and n_train == 0 and split_ratio[0] > 0:
        n_train = 1

    if n_test == 0:
        n_val = n_total - n_train
    else:
        if n_train + n_val + n_test > n_total:
            n_val = max(0, n_total - n_train - n_test)

    train = subjects[:n_train]
    val   = subjects[n_train:n_train + n_val]
    test  = subjects[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def build_loaders(
    lmdb_path: str | Path,
    split_ratio: Tuple[float, float, float],
    batch_size: int,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    shuffle_val:  bool = False,
    logger: Logger = None
):
    lmdb_path = str(lmdb_path)

    all_subjects = get_subject_ids(Path(lmdb_path))
    train_subjects, val_subjects, test_subjects = split_subjects(
        all_subjects, split_ratio=split_ratio, seed=seed
    )
    if logger:
        logger.info(f"[LOADER] Total subjects: {len(all_subjects)}")
        logger.info(f"[LOADER] Train subjects: {len(train_subjects)}")
        logger.info(f"[LOADER] Validation subjects: {len(val_subjects)}")
        logger.info(f"[LOADER] Test subjects: {len(test_subjects)}")

    print(f"[LOADER] Total subjects: {len(all_subjects)}")
    print(f"[LOADER] Train subjects: {len(train_subjects)}")
    print(f"[LOADER] Validation subjects: {len(val_subjects)}")
    print(f"[LOADER] Test subjects: {len(test_subjects)}")

    train_dataset = CustomDataset(lmdb_path, train_subjects)
    val_dataset = CustomDataset(lmdb_path, val_subjects)
    test_dataset = CustomDataset(lmdb_path, test_subjects)
    shuffle_val = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
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





if __name__ == "__main__":
    tueg_path = "/projects/EEG-foundation-model/RECH202/tueg_preprocessed/tueg.lmdb"
    tuab_path = "/projects/EEG-foundation-model/RECH202/tuab_preprocessed/tuab.lmdb"
    physionetmi_path = "/projects/EEG-foundation-model/RECH202/physionetmi_preprocessed/physionetmi.lmdb"
    sleepedfx_path = "/projects/EEG-foundation-model/RECH202/sleepedfx_preprocessed/sleepedfx.lmdb"
    seedv_path = "/projects/EEG-foundation-model/RECH202/seedv_preprocessed/seedv.lmdb"
    # bci2a_path = "/projects/EEG-foundation-model/RECH202/bci2a_preprocessed/bci2a.lmdb"
    bci2a_path_30s = "/projects/EEG-foundation-model/RECH202/data_preprocessed/tests/ws_30s/BCI2A/bci2a.lmdb"
    bci2a_path_10s = "/projects/EEG-foundation-model/RECH202/data_preprocessed/tests/ws_10s/BCI2A/bci2a.lmdb"
    bci2a_path_1s = "/projects/EEG-foundation-model/RECH202/data_preprocessed/tests/ws_1s/BCI2A/bci2a.lmdb"
    faced_path = "/projects/EEG-foundation-model/RECH202/faced_preprocessed/faced.lmdb"

    subject_ids = get_subject_ids(Path(bci2a_path_30s))

    dataset = CustomDataset(
        lmdb_path= bci2a_path_30s,
        subject_ids=subject_ids
    )

    plot_random_eeg_samples(
        ds=dataset,
        out_dir="/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/figures/random_eeg_samples/bci2a_30s",
        n_samples=8,
        n_channels=6,
        seed=42,
        t_len=None,
    )


    # test on tuab
    # train_loader, val_loader, test_loader = build_loaders(
    #     bci2a_path,
    #     split_ratio=(0.8, 0.1, 0.1),
    #     batch_size=1,
    #     seed=42,
    #     num_workers=2,
    #     pin_memory=True,
    #     persistent_workers=False,
    #     shuffle_val=False,
    # )

    # for batch in train_loader:
    #     x,y = batch
    #     x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    #     mean_ch = x_np.mean(axis=-1)          # (C,)
    #     std_ch  = x_np.std(axis=-1)           # (C,)
    #     print(x.shape, y.shape)
    #     print(f"Mean per channel: {mean_ch}")
    #     print(f"Std per channel: {std_ch}")
    #     print(y)
    #     break