import argparse
from pathlib import Path
from yaml import safe_load
from preprocessing import preprocess_dataset
from utils import init_logger, clean_output_dir
from copy import deepcopy



def get_preprocess_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = safe_load(f)
    return config


def main(config: dict = None):
    dataset_name = config.get("dataset_name", None)
    dataset_dir = config[dataset_name].get("path", None)
    logger_dir = config.get("logger_dir", None) + "/" + dataset_name + "_preprocessing"
    log_step = config[dataset_name].get("log_step", 20)
    logger = init_logger(logger_dir, logger_name = dataset_name)
    dataset_cfg = config.get(dataset_name, {})
    preprocessing_params = deepcopy(config.get("preprocessing_params", None)) or {}
    # notch and window_size_sec are dataset-specific (line frequency, task window),
    # so per-dataset values in the YAML override the global preprocessing_params.
    for key in ("notch", "window_size_sec"):
        if key in dataset_cfg:
            preprocessing_params[key] = dataset_cfg[key]
    debug = config.get("debug", False)
    files_ratio = config[dataset_name].get("files_ratio", 1.0)
    out_dir = config.get("out_dir", None) + "/" + dataset_name 
    clean_up = config.get("clean_up", False)

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        if clean_up:
            clean_output_dir(out_dir)
            if logger:
                logger.info(f"Output directory {out_dir} already exists. Cleaning up...")

    preprocess_dataset(
        dataset_name= dataset_name,
        dataset_dir= dataset_dir,
        out_dir= out_dir,
        files_ratio= files_ratio,
        logger= logger,
        debug= debug,
        log_step= log_step,
        preprocessing_config= preprocessing_params,
        dataset_cfg= dataset_cfg,
        )



def get_window_config(dataset: str, config: dict) -> dict:
    base_config = deepcopy(config)
    base_config["dataset_name"] = dataset
    SIZES = [30, 10, 1]
    configs = {}
    for window_size in SIZES:
        window_config = deepcopy(base_config)
        window_config["out_dir"] = f"{config.get('out_dir', None)}/ws_{window_size}s"
        window_config["preprocessing_params"]["window_size_sec"] = window_size
        # main() lets the per-dataset value override preprocessing_params, so set it
        # on the dataset block too to force the swept window size for this dataset.
        window_config[dataset]["window_size_sec"] = window_size
        configs[window_size] = window_config
    return configs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG preprocessing for one dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to preprocess (overrides 'dataset_name' in the config). "
             "Must match a dataset block in the YAML, e.g. TUEG, FACED, SEEDV, ...",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "preprocessing.yaml",
        help="Path to the preprocessing YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = get_preprocess_config(args.config)

    if args.dataset is not None:
        config["dataset_name"] = args.dataset

    dataset_name = config.get("dataset_name", None)
    if dataset_name not in config:
        raise ValueError(
            f"Dataset '{dataset_name}' has no configuration block in {args.config}."
        )

    main(config)