
from pathlib import Path

base = Path("artifacts/data_preprocessing/processed")

print("Processed directory:", base.resolve())
print("Directory exists:", base.exists())

if base.exists():
    files = [path for path in base.rglob("*") if path.is_file()]

    print(f"\nTotal files found: {len(files)}")

    for path in files[:100]:
        print(f"{path}  |  {path.suffix}")

    if not files:
        print("\nNo files found anywhere under processed/.")