import numpy as np
from pathlib import Path

root = Path("artifacts")

for machine in ["machine-2-8", "machine-3-2"]:
    scores = np.load(
        root / f"anomaly_detection/scores/test/{machine}.npy"
    ).reshape(-1)

    labels = np.load(
        root / f"label_alignment/labels/{machine}.npy"
    ).reshape(-1)

    if len(scores) != len(labels):
        print(f"{machine}: score/label length mismatch")
        continue

    print(f"\n{machine}")

    for name, mask in [
        ("Normal labels", labels == 0),
        ("Anomaly labels", labels == 1),
    ]:
        values = scores[mask]

        if len(values):
            print(
                f"{name}: count={len(values)}, "
                f"median={np.median(values):.6g}, "
                f"95th={np.percentile(values, 95):.6g}, "
                f"99th={np.percentile(values, 99):.6g}, "
                f"max={np.max(values):.6g}"
            )

    print("Non-finite scores:", int((~np.isfinite(scores)).sum()))