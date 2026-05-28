#!/bin/bash
# One-off bootstrap for the HSE cHARISMa cluster.
# Run this on the LOGIN node (not via sbatch) — it has internet for pip/conda.
#
# Creates a project Python environment with qiskit + qiskit-aer (GPU build)
# and prints what we got. Idempotent: re-running just verifies the env.

set -euo pipefail

echo "===== ENV DISCOVERY ====="
whoami
hostname
echo "PWD = $(pwd)"
echo "HOME = $HOME"
df -h "$HOME" | head -2

echo "===== PROJECTS / NODE TYPES ====="
mp 2>&1 | head -20 || echo "mp not found"
nodetypes 2>&1 | head -30 || echo "nodetypes not found"
freenodes 2>&1 | head -20 || echo "freenodes not found"

echo "===== AVAILABLE MODULES (python / cuda / cuquant) ====="
module avail 2>&1 | grep -iE "python|anaconda|cuda|cuquant" | head -40 || true

echo "===== LOAD PYTHON ====="
module purge
module load Python/Anaconda_v11.2021 2>/dev/null \
    || module load Python/Anaconda_v10.2019 2>/dev/null \
    || module load Python 2>/dev/null \
    || echo "no Python module — relying on system"

which python
python --version

echo "===== CREATE / ACTIVATE 'cif' CONDA ENV ====="
if ! conda env list | grep -q '^cif '; then
    conda create -y -n cif python=3.11 numpy scipy
fi
source activate cif || conda activate cif

echo "===== INSTALL QISKIT + AER (GPU) ====="
# qiskit-aer-gpu wheels include cuQuantum bindings; PyPI hosts the cu12 build.
pip install --upgrade pip
pip install qiskit==2.3.1 qiskit-aer-gpu-cu12 || pip install qiskit==2.3.1 qiskit-aer-gpu

python -c "import qiskit, qiskit_aer; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"

echo "===== TEST GPU AVAILABILITY ====="
python - <<'PY'
from qiskit_aer import AerSimulator
try:
    sim = AerSimulator(method='statevector', device='GPU')
    print("GPU backend OK:", sim.configuration().description)
    print("available_devices =", getattr(sim.configuration(), 'available_devices', '?'))
except Exception as e:
    print("GPU backend probe FAILED:", e)
    print("(this is fine on the login node — login nodes typically lack GPUs;")
    print(" the real check happens inside the sbatch job.)")
PY

echo "===== BOOTSTRAP DONE ====="
echo "Next step: submit the experiment with"
echo "    sbatch hpc_qaoa.sbatch"
