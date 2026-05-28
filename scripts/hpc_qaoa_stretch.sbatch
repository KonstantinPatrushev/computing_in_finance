#!/bin/bash
# Stretch experiment: QAOA at N=24, 26, 28 on actual GPU (A100/H100).
# Uses the preemptable gpu-ef-quick queue — may be interrupted at any moment
# if higher-priority work arrives. --requeue tells SLURM to re-queue us if
# that happens, so we eventually finish on free GPU time.

#SBATCH --job-name=cif_qaoa_stretch
#SBATCH --account=proj_1804
#SBATCH --output=hpc_qaoa_stretch-%j.log
#SBATCH --error=hpc_qaoa_stretch-%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --requeue

echo "===== JOB INFO ====="
echo "Date         = $(date)"
echo "Hostname     = $(hostname)"
echo "Working dir  = $(pwd)"
echo "Job ID       = $SLURM_JOB_ID"
echo "Allocated GPU= $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "================================="

module purge
module load Python/Anaconda_v11.2021
source activate cif

echo "Python: $(which python)"
python --version
python -c "import qiskit, qiskit_aer, numpy; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__, 'numpy', numpy.__version__)"

# Push AerSimulator(GPU) imports up so we know whether cuQuantum loads here.
python -c "from qiskit_aer import AerSimulator; s = AerSimulator(method='statevector', device='GPU'); print('AerSimulator GPU OK:', s.configuration().description)" \
    || echo "WARN: AerSimulator GPU probe failed — pipeline will use numpy fallback"

mkdir -p results
# 3 N × 1 p × 3 seeds = 9 cells. Keeps within 1 h preempt window.
python -u hpc_qaoa_gpu.py \
    --N 24 26 28 \
    --p 1 \
    --seeds 42 123 7 \
    --shots 4096 \
    --device GPU \
    --out results/hpc_qaoa_stretch.jsonl

echo "===== DONE at $(date) ====="
