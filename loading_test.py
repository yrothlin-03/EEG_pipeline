from loaders import build_loaders


if __name__ == "__main__":
    lmdb_path = "/projects/EEG-foundation-model/RECH202/seedv_preprocessed/seedv.lmdb"
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
    for step, (x, y) in enumerate(train_loader, 1):
        print(x.shape)
        print(y)
        if step >= 3:
            break