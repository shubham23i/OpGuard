
from pathlib import Path
import numpy as np

label_dir = Path("artifacts/label_alignment/labels")
score_dir = Path("artifacts/anomaly_detection/scores/test")

for label_file in sorted(label_dir.glob("*.npy")):
    score_file = score_dir / label_file.name

    if not score_file.exists():
        print(f"{label_file.stem}: MISSING SCORE FILE")
        continue

    labels = np.load(label_file).ravel()
    scores = np.load(score_file).ravel()

    print(
        f"{label_file.stem}: "
        f"labels={len(labels)}, scores={len(scores)}, "
        f"anomaly_rate={labels.mean():.4f}, "
        f"aligned={len(labels) == len(scores)}"
    )
