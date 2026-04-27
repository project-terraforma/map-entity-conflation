from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PipelineConfig:
    """Typed wrapper around the YAML settings file."""

    raw: dict[str, Any]
    root_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(raw=raw, root_dir=config_path.parent.parent.resolve())

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cursor: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def resolve_path(self, value: str | None) -> str | Path | None:
        if not value:
            return None
        if "://" in value:
            return value
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    def require_input_paths(self, dotted_keys: list[str]) -> None:
        missing = [key for key in dotted_keys if not self.get(key)]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required input paths in config: {joined}")

    @property
    def target_crs(self) -> str:
        return str(self.get("target_crs", "EPSG:26913"))

    @property
    def interim_dir(self) -> Path:
        return (self.root_dir / "data" / "interim").resolve()

    @property
    def output_dir(self) -> Path:
        return (self.root_dir / "data" / "outputs").resolve()

    @property
    def figures_dir(self) -> Path:
        return (self.output_dir / "figures").resolve()
