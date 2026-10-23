
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

root = Path("artifacts")

for machine in ["machine-2-8", "machine-3-2"]:
    scores = np.load(
        root / f"anomaly_detection/scores/test/{machine}.npy"
    ).reshape(-1)

    labels = np.load(
        root / f"label_alignment/labels/{machine}.npy"
    ).reshape(-1).astype(int)

    print(f"\n{'=' * 65}")
    print(machine)
    print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")

    if machine == "machine-2-8":
        candidates = [1, 10, 50, 100, 150, 200, 250, 300, 400, 500]
    else:
        candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for threshold in candidates:
        predictions = (scores > threshold).astype(int)

        precision = precision_score(
            labels, predictions, zero_division=0
        )
        recall = recall_score(
            labels, predictions, zero_division=0
        )
        f1 = f1_score(
            labels, predictions, zero_division=0
        )

        print(
            f"{threshold:12.3f} {precision:12.4f} "
            f"{recall:12.4f} {f1:12.4f}"
        )
    