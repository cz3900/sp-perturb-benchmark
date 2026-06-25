# Celcomen offline env (server)

## Build
    conda create -n celcomen python=3.9 -y && conda activate celcomen
    pip install git+https://github.com/stathismegas/celcomen
    # torch + PyG per the repo's pyproject; GPU node03.

## Smoke
    python -c "from celcomen.models.simcomen import Simcomen; print('ok')"

## Notes
- RAW integer counts, NO normalization (the adapter enforces this).
- KO = set_sphex zero on the in-panel guide column; off-panel guides are skipped (option 2).
- Confirm the exact CCE.fit / Simcomen generate signatures against
  Tutorial_Celcomen_on_Xenium.ipynb and the spatial_KO tutorial before the first real run;
  adjust `scripts/celcomen/run_celcomen.py::main` if the API differs.
