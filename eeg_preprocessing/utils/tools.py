import lmdb
from pathlib import Path
import shutil
from typing import Optional, Tuple


def clean_output_dir(out_dir: Path) -> None:
    out_dir = Path(out_dir).expanduser()
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"{out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)


def open_out_file(
    out_dir: Path,
    dataset_name: str,
    file_format: str = "lmdb",
    map_size: int = 1024**3 * 500,
) -> tuple:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_format = file_format.lower()
    if file_format == "lmdb":
        lmdb_path = out_dir / f"{dataset_name.lower()}.lmdb"
        env = lmdb.open(
            str(lmdb_path),
            map_size=int(map_size),
            subdir=False,
            lock=True,         
            readahead=False,    
            meminit=False,
            max_dbs=1,
        )
        return env, lmdb_path

    raise ValueError(f"Unsupported out file format: {file_format}")