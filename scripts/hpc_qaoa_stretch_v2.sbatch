#!/bin/bash
# Stretch v2: push QAOA quality + GPU utilisation.
# - p=2 at N=24 (testing whether deeper QAOA closes the 5.5% gap p=1 had)
# - p=1 at N=26 (larger statevector — 64M complex = 1 GB)
# - warm-start: for p=2, COBYLA starts from previously trained p=1 params
# - reduced restarts (3 instead of 6) to fit within preemptable time budget

#SBATCH --job-name=cif_qaoa_v2
#SBATCH --account=proj_1804
#SBATCH --output=hpc_qaoa_stretch_v2-%j.log
#SBATCH --error=hpc_qaoa_stretch_v2-%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --requeue
# The script writes one JSONL row per completed (N, p, seed) cell and
# automatically skips already-completed cells on restart. So a preempt
# only costs the in-progress cell, not previously finished ones.

echo "===== JOB INFO ====="
echo "Date         = $(date)"
echo "Hostname     = $(hostname)"
echo "Job ID       = $SLURM_JOB_ID"
echo "Allocated GPU= $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "================================="

module purge
module load Python/Anaconda_v11.2021
source activate cif

python --version
python -c "import qiskit, qiskit_aer, numpy; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'numpy', numpy.__version__)"
python -c "from qiskit_aer import AerSimulator; s = AerSimulator(method='statevector', device='GPU'); print('AerSimulator GPU OK:', s.configuration().description)"

mkdir -p results

# Step 1: N=24 p={1,2} — compare p=2 to p=1 with warm-start
python -u hpc_qaoa_gpu.py \
    --N 24 \
    --p 1 2 \
    --seeds 42 123 7 \
    --shots 4096 \
    --device GPU \
    --n-restarts 3 \
    --warm-start \
    --out results/hpc_qaoa_v2_n24.jsonl

# Step 2: N=26 p=1 — push to larger statevector (~1 GB, GPU-bound regime)
python -u hpc_qaoa_gpu.py \
    --N 26 \
    --p 1 \
    --seeds 42 123 7 \
    --shots 4096 \
    --device GPU \
    --n-restarts 3 \
    --out results/hpc_qaoa_v2_n26.jsonl

echo "===== DONE at $(date) ====="
