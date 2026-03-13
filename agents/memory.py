from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    round_id: int
    partner_id: Optional[str]
    event_type: str
    content: str
    outcome: Dict[str, Any]


class MemoryBuffer:
    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        self.items.append(item)

        if len(self.items) > self.max_items:
            self.items.pop(0)

    def get_recent(self, n: int = 5):
        return self.items[-n:]