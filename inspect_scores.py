import numpy as np
from pathlib import Path

root = Path("artifacts")
thresholds = np.load(
    root / "threshold_manager/thresholds.npy",
    allow_pickle=True
).item()

for machine in ["machine-2-8", "machine-3-2"]:
    train = np.load(root / f"anomaly_detection/scores/train/{machine}.npy")
    scores = np.load(root / f"anomaly_detection/scores/test/{machine}.npy")
    labels = np.load(root / f"label_alignment/labels/{machine}.npy")
    predictions = np.load(
        root / f"threshold_manager/predictions/test/{machine}.npy"
    )
    threshold = thresholds[machine]

    print(f"\n{'=' * 45}")
    print(f"Machine: {machine}")
    print(f"Threshold: {threshold:.8g}")
    print("Train score percentiles:", np.percentile(train, [50, 90, 95, 99, 99.9]))
    print("Test score percentiles:", np.percentile(scores, [50, 90, 95, 99, 99.9, 100]))
    print("Positive labels:", int(labels.sum()))
    print("Positive predictions:", int(predictions.sum()))
    print("Scores above threshold:", int((scores > threshold).sum()))