#!/bin/bash
#SBATCH --job-name=cif_qaoa_gpu
#SBATCH --account=proj_1804
#SBATCH --output=hpc_qaoa-%j.log
#SBATCH --error=hpc_qaoa-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --constraint=type_a|type_b|type_c

# Print runtime info up front
echo "===== JOB INFO ====="
echo "Date         = $(date)"
echo "Hostname     = $(hostname)"
echo "Working dir  = $(pwd)"
echo "Job ID       = $SLURM_JOB_ID"
echo "Allocated GPU= $CUDA_VISIBLE_DEVICES"
echo "================================="

# Pick the most recent Python module visible to the user.
# If the cluster ships Anaconda, we activate the project conda env created
# during the bootstrap step (`hpc_bootstrap.sh`).
module purge
module load Python/Anaconda_v11.2021

# Activate the conda env we built in hpc_bootstrap.sh
source activate cif

echo "Python: $(which python)"
python --version
python -c "import qiskit, qiskit_aer; print('qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)"
python -c "from qiskit_aer import AerSimulator; s = AerSimulator(method='statevector', device='GPU'); print('GPU backend OK:', s.configuration().description)" \
    || echo "WARN: GPU backend probe failed"

# Run the experiment
mkdir -p results
python -u hpc_qaoa_gpu.py \
    --N 8 10 12 14 16 18 20 \
    --p 1 2 \
    --seeds 42 123 7 \
    --shots 4096 \
    --device GPU \
    --out results/hpc_qaoa.jsonl

echo "===== DONE at $(date) ====="
