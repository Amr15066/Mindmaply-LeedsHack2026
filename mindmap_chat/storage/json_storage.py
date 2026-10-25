"""
JSON file-based storage for conversation graphs.
Simple, local, easy to debug and inspect.
Atomic writes with file locking to prevent corruption.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from mindmap_chat.models import Mindmap

from .migrations import SCHEMA_VERSION, UnrecognisedDocument, migrate


class JSONStorage:
    """JSON file storage for conversation graphs."""

    def __init__(self, file_path: str = "./data/conversation.json"):
        """
        Initialize storage.
        
        Args:
            file_path: Path to JSON file
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()  # Thread-safe writes

    def save(self, mindmap: Mindmap) -> None:
        """
        Save mindmap to JSON file with atomic write.
        Writes to temp file first, then renames to prevent corruption.
        
        Args:
            mindmap: Mindmap to save
        """
        data = {"schema_version": SCHEMA_VERSION, **mindmap.to_dict()}

        with self._lock:
            # Write to temporary file in same directory (ensures same filesystem)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.file_path.parent,
                prefix=".tmp_",
                suffix=".json"
            )
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Atomic rename (all-or-nothing on most filesystems)
                os.replace(temp_path, self.file_path)
                print(f"[SAVED] {self.file_path}")
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

    def load(self) -> Mindmap:
        """
        Load conversation graph from JSON file (thread-safe).
        
        Returns:
            Loaded Mindmap, or empty mindmap if file doesn't exist
        """
        with self._lock:
            if not self.file_path.exists():
                return Mindmap()
            
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise UnrecognisedDocument("Top-level JSON value is not an object")
                data = migrate(data)
            except (json.JSONDecodeError, UnrecognisedDocument) as exc:
                # Keep the unreadable file: the next save() would overwrite it.
                backup = self.file_path.with_name(
                    f"{self.file_path.name}.unreadable-{int(time.time())}"
                )
                os.replace(self.file_path, backup)
                print(
                    f"[ERROR] Could not read {self.file_path} ({exc}); moved it to {backup} "
                    f"and started an empty mindmap"
                )
                return Mindmap()
            # NewerSchemaError propagates: never silently discard newer data.

        mindmap = Mindmap.from_dict(data)
        for graph in mindmap.graphs.values():
            graph.rebuild_children()
        return mindmap

    def clear(self) -> None:
        """Delete the storage file (thread-safe)."""
        with self._lock:
            if self.file_path.exists():
                os.remove(self.file_path)
                print(f"[CLEARED] {self.file_path}")
