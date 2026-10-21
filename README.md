# Square1 Season 1

My code for the four Square1 Season 1 competitions. One folder per contest:

| Folder | Contest |
|---|---|
| `solar/` | solar-britain |
| `pm25/` | clean-air-usa |
| `emissions/` | emissions-gap |
| `nba/` | nba-margin |

Each folder has `model.py` (features + model) and `run_real.py` (validation + submission files).

## Setup

Python 3.13, CPU only.

```
pip install -r requirements.txt
```

## Running

The contest data isn't included here. Download the train/test CSVs from the contest page, then run from inside the contest folder:

```
cd solar
python run_real.py --train ./data/train.csv --test ./data/test.csv --out ./submissions
```

All three arguments are optional. Defaults are `./data/train.csv`, `./data/test.csv` and `./submissions`, and you can also set `TRAIN_CSV`, `TEST_CSV` and `OUT_DIR` as environment variables.

Each run writes progress to `real_progress.log`, the validation summary to `report_real.json`, and two submission CSVs to the output folder. Runs take a few minutes at most.

`nba/LEAKAGE.md` describes a target leak in the NBA data. It was not used.
