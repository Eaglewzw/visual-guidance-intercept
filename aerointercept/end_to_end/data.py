"""Episode-sharded dataset utilities for end-to-end behavior cloning."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


DATASET_SCHEMA_VERSION = 1
REQUIRED_ARRAYS = (
    "frames",
    "actions",
    "future_position",
    "collision_risk",
    "confidence",
)


def episode_files(data_dir) -> list[Path]:
    paths = sorted((Path(data_dir) / "episodes").glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode shards found under {data_dir}")
    return paths


def load_manifest(data_dir) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    version = int(manifest.get("schema_version", -1))
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported dataset schema {version}; expected "
            f"{DATASET_SCHEMA_VERSION}")
    return manifest


def split_episode_files(paths, validation_fraction: float, seed: int):
    """Split at episode granularity so adjacent frames never leak to val."""
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("at least two episodes are required for train/validation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    validation_count = max(1, int(round(len(paths) * validation_fraction)))
    validation_count = min(validation_count, len(paths) - 1)
    validation_indices = set(order[:validation_count].tolist())
    train = [path for index, path in enumerate(paths)
             if index not in validation_indices]
    validation = [path for index, path in enumerate(paths)
                  if index in validation_indices]
    return train, validation


class EpisodeSequenceDataset(Dataset):
    """Read fixed windows with a bounded cache of decompressed episodes.

    The final window is padded by repeating its last valid state and accompanied
    by a mask.  Frame histories are built inside an episode, never across reset
    boundaries.
    """

    def __init__(self, paths, sequence_length: int, history_frames: int = 2,
                 cache_size: int = 16):
        self.paths = [Path(path) for path in paths]
        self.sequence_length = int(sequence_length)
        self.history_frames = int(history_frames)
        if self.sequence_length < 1 or self.history_frames < 1:
            raise ValueError("sequence_length and history_frames must be positive")
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()
        self._windows = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as shard:
                self._validate_shard(path, shard)
                length = int(shard["frames"].shape[0])
            for start in range(0, length, self.sequence_length):
                self._windows.append((path, start, length))
        if not self._windows:
            raise ValueError("dataset contains no transitions")

    @staticmethod
    def _validate_shard(path, shard) -> None:
        missing = [name for name in REQUIRED_ARRAYS if name not in shard]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        length = int(shard["frames"].shape[0])
        if length < 1:
            raise ValueError(f"{path} is empty")
        if shard["frames"].ndim != 4 or shard["frames"].shape[1] != 3:
            raise ValueError(f"{path}: frames must have shape [T,3,H,W]")
        if shard["frames"].dtype != np.uint8:
            raise ValueError(f"{path}: frames must use uint8 storage")
        if shard["actions"].shape != (length, 4):
            raise ValueError(f"{path}: actions must have shape [T,4]")
        if shard["future_position"].shape != (length, 3):
            raise ValueError(f"{path}: future_position must have shape [T,3]")
        for name in ("collision_risk", "confidence"):
            if shard[name].shape != (length,):
                raise ValueError(f"{path}: {name} must have shape [T]")
        for name in REQUIRED_ARRAYS[1:]:
            if int(shard[name].shape[0]) != length:
                raise ValueError(f"{path}: {name} length does not match frames")

    def __len__(self):
        return len(self._windows)

    def _load(self, path: Path):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(path, allow_pickle=False) as shard:
            arrays = {name: shard[name] for name in REQUIRED_ARRAYS}
        self._cache[key] = arrays
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        path, start, episode_length = self._windows[index]
        arrays = self._load(path)
        valid_length = min(self.sequence_length, episode_length - start)
        raw_indices = np.arange(start, start + valid_length)

        history_indices = []
        for current in raw_indices:
            first = current - self.history_frames + 1
            history_indices.append([
                max(0, first + offset)
                for offset in range(self.history_frames)
            ])
        frames = arrays["frames"][np.asarray(history_indices)]

        result = {"frames": frames}
        for name in REQUIRED_ARRAYS[1:]:
            result[name] = arrays[name][raw_indices]
        mask = np.ones(valid_length, dtype=np.float32)

        padding = self.sequence_length - valid_length
        if padding:
            for name, values in tuple(result.items()):
                repeated = np.repeat(values[-1:], padding, axis=0)
                result[name] = np.concatenate([values, repeated], axis=0)
            mask = np.concatenate([
                mask, np.zeros(padding, dtype=np.float32)])
        result["mask"] = mask
        return result
