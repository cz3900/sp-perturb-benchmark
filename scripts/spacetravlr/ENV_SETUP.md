# SpaceTravLR offline env (server)

SpaceTravLR self-submits Slurm via `spawn_worker`; we DO NOT use it — the runner calls
`setup_ → fit → setup_perturbations → perturb` directly inside an `srun` allocation.

## Build (uv, on the server)
    cd ~ && git clone https://github.com/jishnu-lab/SpaceTravLR && cd SpaceTravLR
    git checkout release
    uv venv ~/.venvs/spacetravlr && source ~/.venvs/spacetravlr/bin/activate
    uv sync                              # installs SpaceTravLR + celloracle + commot + torch
    # GPU: node03 (RTX 3090). Verify: python -c "import torch; print(torch.cuda.is_available())"

## Smoke
    python -c "from SpaceTravLR.spaceship import SpaceShip; print(SpaceShip.load_base_GRN('mouse').shape)"

## Notes
- Single-cell datasets only (Saunders/Shen/Binan/Cheng); Dhainaut (Visium spot) is out of scope.
- `setup_(run_commot=True)` runs CellOracle (TF GRN) + COMMOT (ligand-receptor) — slow; cache per dataset.
- Coverage: only guides in `load_base_GRN(species)` (TF) ∪ CellChat L-R are injectable.
