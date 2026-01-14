import logging
from pathlib import Path
from logging import Logger


def init_logger(
    log_dir: str,
    logger_name: str = "preprocessing",
    level: int = logging.INFO,
    file_mode: str = "w",
) -> Logger:

    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False


    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    log_file = log_dir / f"{logger_name}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger