from pathlib import Path
import mne
import numpy as np
from scipy.io import loadmat

def display_gdf_content(gdf_path: str | Path) -> None:
    gdf_path = Path(gdf_path)
    if not gdf_path.exists():
        raise FileNotFoundError(gdf_path)

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    print("=" * 80)
    print(f"File: {gdf_path}")
    print("=" * 80)

    print("Sampling frequency:", raw.info["sfreq"])
    print("Channels:", raw.info["nchan"])
    print("Duration (s):", raw.n_times / raw.info["sfreq"])

    print("\nChannels:")
    for name, typ in zip(raw.ch_names, raw.get_channel_types()):
        print(f"{name:<15} {typ}")

    data = raw.get_data()
    print("\nData shape:", data.shape)
    print("Data dtype:", data.dtype)

    print("\nPer-channel stats:")
    for ch, m, s in zip(raw.ch_names, data.mean(axis=1), data.std(axis=1)):
        print(f"{ch:<15} mean={m:.3e} std={s:.3e}")

    ann = raw.annotations
    print("\nAnnotations:", len(ann))
    for i, (o, d, desc) in enumerate(zip(ann.onset, ann.duration, ann.description)):
        print(f"{i:03d} onset={o:.3f}s duration={d:.3f}s desc='{desc}'")

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        print("\nEvents:", event_id)
        print(events[:10])
    except Exception as e:
        print("\nEvents: none", e)

    print("=" * 80)




def display_mat_content(mat_path: str | Path) -> None:
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    print(mat['data'])
    print("=" * 80)
    print(f"File: {mat_path}")
    print("=" * 80)

    # Exclure les clés internes MATLAB
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print("Variables in file:")
    for k in keys:
        v = mat[k]
        if isinstance(v, np.ndarray):
            print(f"{k:<20} array shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{k:<20} type={type(v)}")

    # Tentative d’identification des données EEG
    data = None
    for k in keys:
        if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2:
            data = mat[k]
            data_key = k
            break

    if data is None:
        print("\nNo 2D data array found (channels × samples).")
        print("=" * 80)
        return

    print("\nAssumed EEG data variable:", data_key)
    print("Data shape:", data.shape)
    print("Data dtype:", data.dtype)

    # Hypothèse : (channels, samples)
    n_channels, n_samples = data.shape

    print("\nPer-channel stats:")
    for ch in range(n_channels):
        m = data[ch].mean()
        s = data[ch].std()
        print(f"Ch{ch:03d}        mean={m:.3e} std={s:.3e}")

    # Fréquence d’échantillonnage (si présente)
    sfreq = None
    for key in ("fs", "sfreq", "sampling_rate", "Fs"):
        if key in mat:
            sfreq = float(mat[key])
            break

    if sfreq is not None:
        print("\nSampling frequency:", sfreq)
        print("Duration (s):", n_samples / sfreq)
    else:
        print("\nSampling frequency: not found")

    print("=" * 80)



if __name__ == "__main__":
    test_path = "/projects/EEG-foundation-model/BCI-IV/A04E.gdf"
    test_path2 = "/projects/EEG-foundation-model/BCI-IV/A04E.mat"
    if Path(test_path).exists():
        # display_gdf_content(test_path)
        display_mat_content(test_path2)
    else:
        print("Test file not found:", test_path)