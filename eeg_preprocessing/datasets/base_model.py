from abc import ABC, abstractmethod
import mne
from pathlib import Path
from logging import Logger


class PreprocessorModel(ABC):
    def __init__(self, dataset_dir: str | Path, logger: Logger = None):
        self.dataset_dir = Path(dataset_dir)
        self.logger = logger
    
    @abstractmethod
    def get_files(self, ratio: float, seed: int = 42) -> list[Path]:
        """Return a list of file paths in the dataset directory."""
        pass
    
    @abstractmethod
    def load_data(self, file_path: Path) -> mne.io.BaseRaw:
        """Load and return the data from the given file path as an MNE Raw object."""
        pass
    @abstractmethod
    def load_labels(self, file_path: Path, raw: mne.io.BaseRaw = None) -> mne.Annotations:
        """Load and return the labels associated with the given file path."""
        pass
    
    def load(self, file_path: Path) -> tuple[mne.io.BaseRaw, mne.Annotations]:
        """Load and return both the data and labels from the given file path."""
        raw = self.load_data(file_path)
        annotations = self.load_labels(file_path, raw=raw)
        subject_id = self.get_subject_id(file_path)
        return raw, annotations, subject_id
    
    @abstractmethod
    def get_subject_id(self, file_path: Path) -> str:
        """Extract and return the subject ID from the given file path."""
        pass

