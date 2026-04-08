"""Serialization of object states to JSON Lines format.

Each line contains one ObjectState as JSON, enabling streaming reads
and easy integration with pandas or post-processing scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase1.core.schemas import ObjectState
from phase1.utils.log import get_logger

logger = get_logger("export.state_writer")


class StateWriter:
    """Writes object states to a JSONL file, one line per object."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        self._count = 0

    def write_frame_objects(self, objects: list[ObjectState]) -> None:
        for obj in objects:
            self._file.write(json.dumps(obj.to_dict()) + "\n")
            self._count += 1

    def close(self) -> None:
        self._file.close()
        logger.info("StateWriter: %d objects written to %s", self._count, self.output_path.name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
