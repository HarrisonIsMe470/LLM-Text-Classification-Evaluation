"""Download and cache the 20 Newsgroups train/test splits."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_20newsgroups


def main() -> None:
    remove = ("headers", "footers", "quotes")
    train = load_20newsgroups(subset="train", remove=remove)
    test = load_20newsgroups(subset="test", remove=remove)
    print(f"Cached 20 Newsgroups: train={len(train.data)}, test={len(test.data)}")


if __name__ == "__main__":
    main()
