from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    round_id: int
    partner_id: Optional[str]
    event_type: str
    content: str
    outcome: Dict[str, Any]


class HierarchicalMemory:
    def __init__(self, max_recent: int = 20, archive_size: int = 100):
        self.max_recent = max_recent
        self.archive_size = archive_size
        self.recent: List[MemoryItem] = []
        self.archive: List[MemoryItem] = []
        self.reflections: List[str] = [] # High-level summaries

    def add(self, item: MemoryItem):
        self.recent.append(item)
        if len(self.recent) > self.max_recent:
            self.archive.append(self.recent.pop(0))
            if len(self.archive) > self.archive_size:
                self.archive.pop(0)

    def retrieve(self, query: str = None, partner_id: str = None, limit: int = 5) -> List[MemoryItem]:
        """
        Search memory for relevant items by partner_id or keywords.
        """
        candidates = self.recent + self.archive
        
        if partner_id:
            results = [m for m in candidates if m.partner_id == partner_id]
            return results[-limit:]
            
        if query:
            keywords = query.lower().split()
            scored = []
            for m in candidates:
                score = sum(1 for kw in keywords if kw in m.content.lower())
                if score > 0:
                    scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for score, m in scored[:limit]]
            
        return self.recent[-limit:]

    def get_recent(self, limit: int = 5) -> List[MemoryItem]:
        """Compatibility method for prompt builder and legacy tests."""
        return self.recent[-limit:]

    def get_full_context(self, limit: int = 10) -> List[MemoryItem]:

        """Provides a mix of recent and significant events."""
        return (self.archive + self.recent)[-limit:]


class MemoryBuffer(HierarchicalMemory):
    """Compatibility alias for legacy tests."""
    def __init__(self, max_items: int = 50):
        super().__init__(max_recent=max_items)