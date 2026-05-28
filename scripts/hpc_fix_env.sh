#!/bin/bash
# Fix the qiskit / qiskit-aer-gpu version mismatch in the cif conda env.
#
# After hpc_bootstrap.sh we had qiskit==2.3.1 + qiskit-aer-gpu==0.15.1 — these
# are incompatible (aer 0.15 was built against qiskit 1.x). The cHARISMa PyPI
# proxy doesn't ship `qiskit-aer-gpu-cu12`, so we have to align on the 1.x
# stack: numpy<2 + qiskit==1.4.x + qiskit-aer-gpu==0.15.1.

set -euo pipefail

echo "===== ACTIVATE ENV ====="
module purge
module load Python/Anaconda_v11.2021
source activate cif
which python
python --version

echo "===== PIN numpy<2 (qiskit 1.x requires it) ====="
# Use the conda index for numpy since pip won't find a binary <2 here.
conda install -y -n cif 'numpy<2' || pip install 'numpy<2'

echo "===== DOWNGRADE qiskit TO 1.4.x ====="
pip install --force-reinstall --no-deps 'qiskit==1.4.2'

echo "===== VERIFY ====="
python <<'PY'
import sys
print(f"python = {sys.version}")
import numpy as np
print(f"numpy = {np.__version__}")
import qiskit
print(f"qiskit = {qiskit.__version__}")
import qiskit_aer
print(f"qiskit_aer = {qiskit_aer.__version__}")

# These are the four classes the experiment script needs
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
print("ALL IMPORTS OK")

# Statevector smoke test (CPU is fine here — login node has no GPU)
qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
sv = Statevector.from_instruction(qc)
print(f"3-qubit GHZ statevector norm = {abs(sv.data).sum():.4f}")
PY

echo "===== FIX DONE ====="
