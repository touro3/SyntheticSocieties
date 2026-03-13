import json
from pathlib import Path


class EventLogger:
    def __init__(self, output_path: str, overwrite: bool = False) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and self.output_path.exists():
            self.output_path.unlink()

    def log_event(self, payload: dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")