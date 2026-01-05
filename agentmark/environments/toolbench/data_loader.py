"""ToolBench data loader.
Responsibilities: read ToolBench/StableToolBench instruction files and provide an iterable task stream."""

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional


class ToolBenchDataLoader:
    """Simple task iterator supporting subset/shuffle/limit."""

    def __init__(
        self,
        data_root: Path,
        split: str = "G1",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed
        self.tasks: List[dict] = self._load_tasks()
        self._cursor = 0

    def _resolve_split_path(self) -> Path:
        """Support data/ and data_example/ layouts."""
        # instruction/<split>_query.json
        candidate = self.data_root / "instruction" / f"{self.split}_query.json"
        if candidate.exists():
            return candidate

        # Direct file
        candidate_alt = self.data_root / f"{self.split}_query.json"
        if candidate_alt.exists():
            return candidate_alt

        # test_instruction/<split>.json
        candidate_test = self.data_root / f"{self.split}.json"
        if candidate_test.exists():
            return candidate_test
            
        candidate_test_instr = self.data_root / "test_instruction" / f"{self.split}.json"
        if candidate_test_instr.exists():
            return candidate_test_instr

        raise FileNotFoundError(
            f"Cannot find {self.split} instruction file; check {self.data_root}/instruction/{self.split}_query.json"
        )

    def _load_tasks(self) -> List[dict]:
        path = self._resolve_split_path()
        with open(path, "r") as f:
            data = json.load(f)

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(data)

        if self.limit is not None:
            data = data[: self.limit]
        return data

    def __iter__(self) -> Iterator[dict]:
        self._cursor = 0
        return iter(self.tasks)

    def __len__(self) -> int:  # noqa: D401
        return len(self.tasks)
