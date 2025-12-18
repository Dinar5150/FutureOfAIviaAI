import glob
import os
from typing import List


def find_datasets(root: str = ".") -> List[str]:
    """Return all SemanticGraph_*.pkl paths under root."""
    pattern = os.path.join(root, "SemanticGraph_*.pkl")
    return sorted(glob.glob(pattern))
