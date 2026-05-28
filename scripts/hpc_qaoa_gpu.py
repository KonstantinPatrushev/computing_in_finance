"""Self-contained GPU QAOA for cardinality-constrained Markowitz.

Designed to run on the HSE cHARISMa cluster (Aer GPU via cuQuantum) without
any project-local dependencies. Imports only: numpy, scipy, qiskit, qiskit-aer.

For each (N, p, seed) it:
  1. Generates a synthetic dense-random Markowitz instance.
  2. Builds a binary-inclusion QUBO with a cardinality penalty.
  3. Translates to an Ising Hamiltonian and a manual QAOA(p) ansatz.
  4. Trains (β, γ) on AerSimulator statevector (GPU if available).
  5. Samples ``shots`` measurements from the final state.
  6. Decodes samples to portfolios, computes E[obj], P(valid),
     and the best-of-shots subset objective.
  7. Compares to the brute-force optimum (subset enumeration over C(N, K)).

Output: a single JSONL file with one row per (N, p, seed, device).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

# qiskit_aer is optional — falls back to pure-Statevector sampling if the
# GPU/cuQuantum native library is incompatible with the host glibc (a common
# situation on older HPC login nodes).
try:
    from qiskit_aer import AerSimulator  # noqa: F401
    HAS_AER = True
except Exception as _aer_exc:
    print(f"[INFO] qiskit_aer unavailable ({_aer_exc.__class__.__name__}: {_aer_exc}); "
          "falling back to pure-numpy Statevector sampling", flush=True)
    AerSimulator = None
    HAS_AER = False


# ---------- problem ----------

def make_problem(n: int, K: int, seed: int = 42, risk_aversion: float = 2.0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return {"mu": mu, "sigma": sigma, "n": n, "K": K, "lam": risk_aversion}


def objective_value(weights: np.ndarray, problem: dict) -> float:
    """Markowitz objective: -μᵀw + (λ/2) wᵀΣw."""
    mu, sigma, lam = problem["mu"], problem["sigma"], problem["lam"]
    return float(-mu @ weights + 0.5 * lam * weights @ sigma @ weights)


def brute_force_subsets(problem: dict) -> dict:
    """Enumerate all size-K subsets with equal weights, return the best."""
    n, K = problem["n"], problem["K"]
    best = None
    for subset in combinations(range(n), K):
        w = np.zeros(n)
        w[list(subset)] = 1.0 / K
        obj = objective_value(w, problem)
        if best is None or obj < best["objective"]:
            best = {"subset": list(subset), "objective": obj}
    return best


# ---------- QUBO → Ising (Pauli dict) ----------

def qubo_to_ising(problem: dict, cardinality_penalty: float | None = None):
    n, K, lam = problem["n"], problem["K"], problem["lam"]
    mu, sigma = problem["mu"], problem["sigma"]

    if cardinality_penalty is None:
        obj_scale = float(np.abs(mu).max()) / K + lam * float(np.abs(sigma).max()) / (K * K)
        cardinality_penalty = 100.0 * max(obj_scale, 1e-6)

    Q = np.zeros((n, n))
    linear = np.zeros(n)

    for i in range(n):
        linear[i] += -mu[i] / K
        Q[i, i] += lam * sigma[i, i] / (2 * K * K)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += lam * sigma[i, j] / (K * K)

    for i in range(n):
        linear[i] += -2.0 * cardinality_penalty * K
        Q[i, i] += cardinality_penalty
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * cardinality_penalty

    constant = cardinality_penalty * K * K

    pauli_coeffs: dict[str, float] = {}
    offset = constant

    def add(key: str, c: float) -> None:
        pauli_coeffs[key] = pauli_coeffs.get(key, 0.0) + c

    for i in range(n):
        offset += 0.5 * linear[i]
        z = ["I"] * n
        z[n - 1 - i] = "Z"
        add("".join(z), -0.5 * linear[i])

    for i in range(n):
        offset += 0.5 * Q[i, i]
        z = ["I"] * n
        z[n - 1 - i] = "Z"
        add("".join(z), -0.5 * Q[i, i])

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if qij == 0.0:
                continue
            offset += qij / 4.0
            z_i = ["I"] * n; z_i[n - 1 - i] = "Z"
            z_j = ["I"] * n; z_j[n - 1 - j] = "Z"
            zz = ["I"] * n; zz[n - 1 - i] = "Z"; zz[n - 1 - j] = "Z"
            add("".join(z_i), -qij / 4.0)
            add("".join(z_j), -qij / 4.0)
            add("".join(zz), +qij / 4.0)

    return pauli_coeffs, offset


# ---------- QAOA ansatz ----------

def qaoa_circuit(pauli_coeffs: dict, n_qubits: int, params: np.ndarray, p: int,
                 measure: bool = True) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    for i in range(n_qubits):
        qc.h(i)

    betas = params[:p]
    gammas = params[p:]

    for layer in range(p):
        gamma = gammas[layer]
        for paulistr, coeff in pauli_coeffs.items():
            if abs(coeff) < 1e-12:
                continue
            z_qubits = [n_qubits - 1 - idx for idx, ch in enumerate(paulistr) if ch == "Z"]
            if not z_qubits:
                continue
            if len(z_qubits) == 1:
                qc.rz(2 * gamma * coeff, z_qubits[0])
            elif len(z_qubits) == 2:
                a, b = sorted(z_qubits)
                qc.cx(a, b)
                qc.rz(2 * gamma * coeff, b)
                qc.cx(a, b)
            else:
                raise ValueError("only 1- and 2-body Z terms supported")

        beta = betas[layer]
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    if measure:
        qc.measure(qr, cr)
    return qc


def train_qaoa(pauli_coeffs: dict, n_qubits: int, p: int, seed: int,
               n_restarts: int = 6, device: str = "GPU",
               warm_init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Optimise (β, γ) by minimising <H_cost> on the simulator backend.

    If ``device == "GPU"`` and qiskit_aer is importable, training runs through
    AerSimulator with cuQuantum statevector (each objective call costs a
    transpile + GPU statevector). Otherwise falls back to pure-numpy
    Statevector.from_instruction.

    If ``warm_init`` is provided (length 2p), use it as the seed of restart 0.
    The remaining ``n_restarts - 1`` use random init. Useful for layerwise
    QAOA training: train p, freeze, extend to p+1 with previous params.
    """
    paulis = list(pauli_coeffs.keys())
    coeffs = [pauli_coeffs[k] for k in paulis]
    cost_op = SparsePauliOp(paulis, coeffs=coeffs)

    rng = np.random.default_rng(seed)

    # Decide once whether we can use Aer GPU; probe it with a tiny circuit
    aer_sim = None
    if HAS_AER and device == "GPU":
        try:
            aer_sim = AerSimulator(method="statevector", device="GPU")
            probe = QuantumCircuit(2)
            probe.h(0); probe.cx(0, 1); probe.save_statevector()
            aer_sim.run(probe).result()
            print(f"  [GPU] AerSimulator(GPU) probe OK — using GPU statevector for training", flush=True)
        except Exception as exc:
            print(f"  [GPU] probe failed ({exc.__class__.__name__}); falling back to numpy", flush=True)
            aer_sim = None

    def objective(params):
        qc = qaoa_circuit(pauli_coeffs, n_qubits, params, p, measure=False)
        if aer_sim is not None:
            qc_save = qc.copy()
            qc_save.save_statevector()
            result = aer_sim.run(qc_save).result()
            sv = Statevector(result.data(0)["statevector"])
        else:
            sv = Statevector.from_instruction(qc)
        return float(np.real(sv.expectation_value(cost_op)))

    best_params = None
    best_val = float("inf")
    for restart in range(n_restarts):
        if restart == 0 and warm_init is not None and len(warm_init) == 2 * p:
            x0 = np.asarray(warm_init, dtype=float)
        else:
            x0 = rng.uniform(0, 2 * np.pi, size=2 * p)
        try:
            res = minimize(objective, x0, method="COBYLA",
                           options={"maxiter": 200, "rhobeg": 0.5})
        except Exception as exc:
            print(f"  restart {restart} failed: {exc}")
            continue
        if res.fun < best_val:
            best_val = float(res.fun)
            best_params = res.x.copy()
    return best_params, best_val


def sample_and_score(qc: QuantumCircuit, n_qubits: int, problem: dict,
                     shots: int, device: str, seed: int = 42) -> dict:
    """Sample from a final QAOA circuit and decode counts.

    Prefers ``qiskit_aer`` if available (GPU statevector via cuQuantum).
    Falls back to pure-numpy sampling from a Statevector when aer cannot be
    imported on the host (e.g. glibc mismatch with libcustatevec on older
    login nodes — sbatch will run on a newer compute node, but we want the
    code to be runnable even on the login server for sanity checks).
    """
    if HAS_AER:
        sim = AerSimulator(method="statevector", device=device)
        job = sim.run(qc, shots=shots)
        counts = job.result().get_counts()
    else:
        # Strip the final measurement to recover the pre-measurement state,
        # then sample shots from |psi|^2 with numpy.
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        probs = np.abs(sv.data) ** 2
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(probs), size=shots, p=probs)
        counts = {}
        for i in idx:
            bs = format(int(i), f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
    total = sum(counts.values())

    K = problem["K"]
    e_obj = 0.0
    p_valid = 0.0
    best_obj = float("inf")
    best_bitstring = None
    for bs, c in counts.items():
        prob = c / total
        bits = [int(ch) for ch in bs[::-1]][:n_qubits]
        x = np.array(bits)
        if int(x.sum()) == K:
            w = x / K
            obj = objective_value(w, problem)
            e_obj += prob * obj
            p_valid += prob
            if obj < best_obj:
                best_obj = obj
                best_bitstring = bs
    return {
        "expected_obj": e_obj,
        "p_valid": p_valid,
        "best_obj": best_obj,
        "best_bitstring": best_bitstring,
        "n_unique_bitstrings": len(counts),
    }


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--p", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--K-ratio", type=float, default=0.25)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU", "PYTHON"])
    parser.add_argument("--n-restarts", type=int, default=6,
                        help="Number of COBYLA restarts per (N, p, seed) cell")
    parser.add_argument("--warm-start", action="store_true",
                        help="For p>=2, initialise from previously trained p-1 params"
                             " (β extended with (β[-1], β[-1]), γ with (γ[-1], γ[-1])).")
    parser.add_argument("--out", type=Path, default=Path("results/hpc_qaoa.jsonl"))
    args = parser.parse_args()

    # Test simulator availability up front
    if HAS_AER:
        try:
            sim = AerSimulator(method="statevector", device=args.device)
            cfg = sim.configuration()
            print(f"backend: {cfg.description}  available_devices={getattr(cfg, 'available_devices', '?')}",
                  flush=True)
        except Exception as e:
            print(f"WARN: device={args.device} unavailable on Aer; falling back to CPU. {e}", flush=True)
            args.device = "CPU"
    else:
        print(f"[INFO] running in pure-numpy Statevector mode (qiskit_aer unavailable)", flush=True)
        args.device = "PYTHON"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing rows to {args.out}", flush=True)

    # Checkpoint: if the JSONL already has rows, skip the (N, p, seed)
    # combinations already completed. Useful for preemptable queues —
    # SLURM --requeue will re-run the script from scratch but we pick up
    # where we left off.
    completed: set[tuple[int, int, int]] = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r["N"], r["p"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        if completed:
            print(f"  resuming: {len(completed)} cells already in {args.out}, will skip", flush=True)

    for N in args.N:
        K = max(2, int(round(args.K_ratio * N)))
        for seed in args.seeds:
            problem = make_problem(N, K, seed=seed)
            t0 = time.perf_counter()
            bf = brute_force_subsets(problem)
            bf_time = time.perf_counter() - t0
            print(f"\n=== N={N} K={K} seed={seed}  BF_opt={bf['objective']:.5f}  BF_time={bf_time:.2f}s ===", flush=True)

            pauli_coeffs, offset = qubo_to_ising(problem)
            prev_params = None
            for p in args.p:
                if (N, p, seed) in completed:
                    print(f"  p={p}: already done in checkpoint, skipping", flush=True)
                    continue
                warm = None
                if args.warm_start and prev_params is not None and len(prev_params) == 2 * (p - 1):
                    # Layerwise: extend [β_1..β_{p-1}, γ_1..γ_{p-1}] to length 2p
                    pm1 = len(prev_params) // 2
                    betas_prev = prev_params[:pm1]
                    gammas_prev = prev_params[pm1:]
                    warm = np.concatenate([betas_prev, [betas_prev[-1]], gammas_prev, [gammas_prev[-1]]])
                    print(f"  warm-start p={p} init from previously trained p={p-1}", flush=True)
                t0 = time.perf_counter()
                params, ising_min = train_qaoa(
                    pauli_coeffs, N, p, seed=seed, device=args.device,
                    n_restarts=args.n_restarts, warm_init=warm,
                )
                train_time = time.perf_counter() - t0
                prev_params = params.copy()
                print(f"  p={p}: trained in {train_time:.2f}s  ising_min={ising_min:.4f}", flush=True)

                qc = qaoa_circuit(pauli_coeffs, N, params, p, measure=True)
                t0 = time.perf_counter()
                metrics = sample_and_score(qc, N, problem, shots=args.shots,
                                           device=args.device, seed=seed)
                sample_time = time.perf_counter() - t0
                print(
                    f"    sample: E[obj]={metrics['expected_obj']:.5f} P_valid={metrics['p_valid']:.3f}"
                    f" best={metrics['best_obj']:.5f} ({metrics['best_bitstring']}) in {sample_time:.2f}s",
                    flush=True,
                )

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "N": N, "K": K, "p": p, "seed": seed,
                    "device": args.device, "shots": args.shots,
                    "bf_optimum": bf["objective"],
                    "bf_subset": bf["subset"],
                    "bf_time_s": bf_time,
                    "qaoa_train_time_s": train_time,
                    "qaoa_sample_time_s": sample_time,
                    "qaoa_params": params.tolist(),
                    "qaoa_ising_min": ising_min,
                    "ising_offset": offset,
                    "expected_obj": metrics["expected_obj"],
                    "p_valid": metrics["p_valid"],
                    "best_obj": metrics["best_obj"],
                    "best_bitstring": metrics["best_bitstring"],
                    "approx_ratio_to_bf": (
                        metrics["best_obj"] / bf["objective"]
                        if bf["objective"] < 0
                        else float("nan")
                    ),
                    "gap_to_bf_pct": (
                        100.0 * (metrics["best_obj"] - bf["objective"]) / abs(bf["objective"])
                        if metrics["best_obj"] != float("inf")
                        else float("inf")
                    ),
                }
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    print(f"\ndone. wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
