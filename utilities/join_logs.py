import pandas as pd
from pathlib import Path
# since the design allows to load a trained model to continue training, this file is particularly helpful to join the resulting split logs.
# EDIT THESE: run folders (in correct chronological order!)
RUN_DIRS = [
    Path("../PATH"),
    Path("../PATH")
]
NAME = RUN_DIRS[0].name
OUT_DIR = Path("../logs/joined_run/NAME")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# columns to offset if they exist
OFFSET_COLS = ["TRAIN_EPOCH", "frame_idx", "update_frame"]

def read_tsv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", dtype=str, engine="python")

def write_tsv(df: pd.DataFrame, p: Path) -> None:
    df.to_csv(p, sep="\t", index=False)

# Determine common filenames across the run dirs
common_files = None
for d in RUN_DIRS:
    files = {f.name for f in d.glob("*.txt")}
    common_files = files if common_files is None else (common_files & files)

if not common_files:
    raise FileNotFoundError("No common .txt files found across the provided folders.")

for fname in sorted(common_files):
    dfs = []
    offsets = {c: 0 for c in OFFSET_COLS}

    for run_dir in RUN_DIRS:
        fpath = run_dir / fname
        df = read_tsv(fpath)

        # apply offsets for any known counter columns in this file
        for col in OFFSET_COLS:
            if col in df.columns:
                # convert safely to int for offsetting
                df[col] = df[col].astype(int) + offsets[col]
                offsets[col] = int(df[col].max())

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out_path = OUT_DIR / fname
    write_tsv(merged, out_path)
    print(f"merged -> {out_path}")

print(f"\nDone. Merged logs in: {OUT_DIR}")