from pathlib import Path
import pandas as pd

def unwrap_epoch_series(s: pd.Series) -> pd.Series:
    e = s.astype(int).to_list()
    out = []
    offset = 0
    prev = None
    for v in e:
        if prev is not None and v < prev:
            offset += prev
        out.append(v + offset)
        prev = v
    return pd.Series(out, index=s.index)

def fix_batch_data_epochs(run_folder: str | Path, filename: str = "batch_data.txt", col: str = "train_epoch") -> None:
    run_folder = Path(run_folder)
    path = run_folder / filename
    df = pd.read_csv(path, sep="\t")
    df[col] = unwrap_epoch_series(df[col])
    df.to_csv(path, sep="\t", index=False)

fix_batch_data_epochs(r"./PATH") #target file path here