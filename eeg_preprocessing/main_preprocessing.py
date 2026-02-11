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
    preprocessing_params = config.get("preprocessing_params", None)
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
        preprocessing_config= preprocessing_params
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
        configs[window_size] = window_config
    return configs



if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/EEG_preprocessing_TELECOM_PARIS/eeg_preprocessing/configs/preprocessing.yaml")
    config = get_preprocess_config(config_path)
    # config["dataset_name"] = "BCI2A"
    # window_configs = get_window_config("BCI2A", config)
    # for ws, ws_config in window_configs.items():
    #     print(f"Starting preprocessing for window size: {ws} seconds")
    #     main(ws_config)
    #     print(f"Finished preprocessing for window size: {ws} seconds")

    main(config)