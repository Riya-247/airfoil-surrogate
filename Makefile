.PHONY: install data experimental train train-aoa optimise validate all test smoke clean

# ---- real data (needs internet access to data.isir.upmc.fr and tmbwg.github.io) ----
install:
	pip install -e ".[airfrans,dev]"

data:
	python scripts/01_build_dataset.py --download --root data/airfrans_raw --out data/airfrans.csv

experimental:
	python scripts/fetch_experimental.py

train:
	python scripts/02_train.py --data data/airfrans.csv --task full

train-aoa:
	python scripts/02_train.py --data data/airfrans.csv --task aoa

optimise:
	python scripts/03_optimise.py --data data/airfrans.csv --task full --model svr

validate:
	python scripts/04_validate_naca0012.py --data data/airfrans.csv --task full

all: data experimental train train-aoa optimise validate

# ---- code test on synthetic stand-in data (outputs go to results/synthetic, never a result) ----
smoke:
	python scripts/01_build_dataset.py --synthetic 1000 --out data/synthetic.csv
	python scripts/02_train.py --data data/synthetic.csv --task full --fast
	python scripts/03_optimise.py --data data/synthetic.csv --task full
	python scripts/04_validate_naca0012.py --data data/synthetic.csv --task full

test:
	python -m pytest -q tests

clean:
	rm -rf results/synthetic models/synthetic data/synthetic.csv
