"""Search result domain models."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    item_id: str
    score: float
    metadata: dict[str, Any]
