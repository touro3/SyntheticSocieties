import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.tracker import rebuild_tracker


def main() -> None:
    rebuild_tracker()
    print("Tracker rebuilt successfully.")


if __name__ == "__main__":
    main()
