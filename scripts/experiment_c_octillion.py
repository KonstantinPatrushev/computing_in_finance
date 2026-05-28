"""Experiment C: QAOA portfolio optimisation on the Bauman Octillion SnowDrop 4q chip.

End-to-end pipeline:

1. Build a 4-asset Markowitz problem with cardinality K=2 using a synthetic
   dataset (small enough that brute force enumerates all 6 valid subsets).
2. Express the binary-inclusion QUBO as an Ising Hamiltonian (ZZ + Z terms).
3. Build a QAOA(p) ansatz of depth p in Qiskit using only the SnowDrop 4q
   basis gates (rx, ry, rz, cz) — the cost and mixer layers are explicit so
   we can transpile cleanly onto the star coupling map (q2 is centre).
4. Train (β, γ) classically on a noiseless statevector simulator (COBYLA).
5. Build the final parameter-bound circuit, transpile to native basis +
   coupling map, and send to:
     - Octillion local emulator (Snowdrop 4q with noise model)
     - Octillion remote chip (real quantum execution)
6. Compare the resulting counts to brute-force optimum and Tabu sampler.

Outputs everything as JSON to ``results/experiment_c_octillion.json``.
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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_tabu_selection


# ---------- problem construction ----------

def make_problem(n: int = 4, K: int = 2, seed: int = 42) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.08, 0.20, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.01 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=K,
    )


def brute_force_subsets(problem: PortfolioProblem, K: int) -> list[dict]:
    """Enumerate all size-K subsets, score each by equal-weight objective."""
    results = []
    n = problem.n
    for subset in combinations(range(n), K):
        w = np.zeros(n)
        w[list(subset)] = 1.0 / K
        obj = problem.objective_value(w)
        results.append({"subset": list(subset), "weights": w.tolist(), "objective": obj})
    return sorted(results, key=lambda r: r["objective"])


# ---------- binary-inclusion QUBO → Ising ----------

def qubo_to_ising(problem: PortfolioProblem, K: int, lambda_card: float | None = None) -> tuple[dict, float]:
    """Return (pauli_dict, offset) for H = -(1/K)μᵀx + (λ/(2K²))xᵀΣx + λ_c(Σx-K)².

    Maps binary x_i ∈ {0,1} to Pauli Z_i ∈ {+1,-1} via x_i = (1 - Z_i)/2.
    ``pauli_dict`` is { 'IZ..Z': coeff } with the convention Qiskit uses
    (rightmost char = qubit 0).
    """
    n = problem.n
    mu = problem.mu.copy()
    sigma = problem.sigma.copy()
    lam = problem.risk_aversion
    if lambda_card is None:
        obj_scale = float(np.abs(mu).max()) / K + lam * float(np.abs(sigma).max()) / (K * K)
        lambda_card = 100.0 * max(obj_scale, 1e-6)

    # Quadratic coefficient matrix on x:  Q_ii = -μ_i/K + (λ/(2K²))Σ_ii + λ_c (offset terms below)
    # Σ_ij quadratic + cardinality squared cross-terms
    # Expand λ_c (Σx − K)² = λ_c (Σ_i x_i² + 2Σ_{i<j} x_i x_j − 2K Σ_i x_i + K²)
    Q = np.zeros((n, n))
    linear = np.zeros(n)

    # Markowitz part
    for i in range(n):
        linear[i] += -mu[i] / K
        Q[i, i] += lam * sigma[i, i] / (2 * K * K)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += lam * sigma[i, j] / (K * K)  # absorbed factor of 2 from symmetric sum

    # Cardinality penalty
    for i in range(n):
        linear[i] += -2.0 * lambda_card * K          # from -2K Σx_i
        Q[i, i] += lambda_card                       # from Σ x_i² (binary)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * lambda_card             # from cross-term 2 x_i x_j

    constant = lambda_card * K * K

    # Substitute x_i = (1 − Z_i)/2:
    #   x_i = 1/2 − Z_i/2
    #   x_i² = x_i (binary)
    #   x_i x_j = (1 − Z_i − Z_j + Z_i Z_j) / 4
    # Pauli coefficients:
    pauli_coeffs: dict[str, float] = {}
    offset = constant

    # Linear: c_i x_i = c_i (1/2 − Z_i/2)
    for i in range(n):
        offset += 0.5 * linear[i]
        z = ["I"] * n
        z[n - 1 - i] = "Z"  # Qiskit reverses qubit order in strings
        key = "".join(z)
        pauli_coeffs[key] = pauli_coeffs.get(key, 0.0) - 0.5 * linear[i]

    # Diagonal quadratic: Q_ii x_i² == Q_ii x_i (binary)
    for i in range(n):
        offset += 0.5 * Q[i, i]
        z = ["I"] * n
        z[n - 1 - i] = "Z"
        key = "".join(z)
        pauli_coeffs[key] = pauli_coeffs.get(key, 0.0) - 0.5 * Q[i, i]

    # Off-diagonal: Q_ij x_i x_j = Q_ij/4 (1 − Z_i − Z_j + Z_i Z_j)
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if qij == 0.0:
                continue
            offset += qij / 4.0
            z_i = ["I"] * n; z_i[n - 1 - i] = "Z"
            z_j = ["I"] * n; z_j[n - 1 - j] = "Z"
            zz = ["I"] * n; zz[n - 1 - i] = "Z"; zz[n - 1 - j] = "Z"
            pauli_coeffs[("".join(z_i))] = pauli_coeffs.get("".join(z_i), 0.0) - qij / 4.0
            pauli_coeffs[("".join(z_j))] = pauli_coeffs.get("".join(z_j), 0.0) - qij / 4.0
            pauli_coeffs[("".join(zz))] = pauli_coeffs.get("".join(zz), 0.0) + qij / 4.0

    return pauli_coeffs, offset


# ---------- QAOA ansatz ----------

def qaoa_circuit(pauli_coeffs: dict, n_qubits: int, params: np.ndarray, p: int) -> QuantumCircuit:
    """Build a QAOA(p) ansatz manually with explicit cost and mixer layers."""
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    # Initial state: |+>^n via Ry(pi/2) on each qubit (avoids Hadamard, native basis)
    for i in range(n_qubits):
        qc.ry(np.pi / 2, i)

    betas = params[:p]
    gammas = params[p:]

    for layer in range(p):
        gamma = gammas[layer]
        # Cost layer: exp(-i gamma H_C)
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
                # Implement exp(-i gamma coeff Z_a Z_b) via CZ-based decomposition
                qc.cz(a, b)
                qc.rz(2 * gamma * coeff, b)
                qc.cz(a, b)
            else:
                raise ValueError("only 1- and 2-Z terms supported")

        # Mixer layer: exp(-i beta sum_i X_i) -> Rx(2 beta) per qubit
        beta = betas[layer]
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    qc.measure(qr, cr)
    return qc


def expected_objective_from_counts(counts: dict, problem: PortfolioProblem, K: int) -> tuple[float, dict]:
    """Compute E[objective] over the measurement distribution, also return
    P(valid cardinality) and best-found objective among sampled bitstrings."""
    n = problem.n
    total = sum(counts.values())
    if total == 0:
        return float("inf"), {"p_valid": 0.0, "best": float("inf")}
    e_obj = 0.0
    p_valid = 0.0
    best_obj = float("inf")
    best_bitstring = None
    for bs, p in counts.items():
        prob = p / total
        # Qiskit returns bitstrings with rightmost = qubit 0
        bits = [int(c) for c in bs[::-1]]
        if len(bits) < n:
            bits = bits + [0] * (n - len(bits))
        x = np.array(bits[:n])
        # Decode as binary-inclusion: w = x/K only if Σx == K
        if int(x.sum()) == K:
            w = x / K
            obj = problem.objective_value(w)
            e_obj += prob * obj
            p_valid += prob
            if obj < best_obj:
                best_obj = obj
                best_bitstring = bs
        else:
            e_obj += prob * 0.0  # penalty-free outside; we report p_valid separately
    return e_obj, {"p_valid": p_valid, "best_obj": best_obj, "best_bitstring": best_bitstring}


def train_qaoa(pauli_coeffs: dict, problem: PortfolioProblem, K: int, p: int = 1, seed: int = 42) -> np.ndarray:
    """Train (β, γ) on noiseless statevector. Returns optimal params of length 2p."""
    rng = np.random.default_rng(seed)
    n = problem.n

    # Convert pauli_coeffs dict to SparsePauliOp for statevector evaluation
    paulis = list(pauli_coeffs.keys())
    coeffs = [pauli_coeffs[p] for p in paulis]
    cost_op = SparsePauliOp(paulis, coeffs=coeffs)

    def objective(params):
        qc = qaoa_circuit(pauli_coeffs, n, params, p)
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        return float(np.real(sv.expectation_value(cost_op)))

    best_params = None
    best_val = float("inf")
    for trial in range(8):
        x0 = rng.uniform(0, 2 * np.pi, size=2 * p)
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": 200, "rhobeg": 0.5})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    return best_params, best_val


# ---------- Octillion run ----------

def run_on_octillion(circuit: QuantumCircuit, backend, shots_log2: int, project: str, poll_timeout: int = 300) -> dict:
    """Submit to Octillion backend and wait for completion. Returns counts as raw dict."""
    job_id = backend.run(circuit, shots=shots_log2, project=project)
    print(f"  submitted: {job_id}")
    t0 = time.time()
    last_status = ""
    while time.time() - t0 < poll_timeout:
        job = _CLIENT.job(job_id)
        if str(job.status) != last_status:
            print(f"  [{time.time()-t0:5.1f}s] status={job.status}")
            last_status = str(job.status)
        if str(job.status) in ("COMPLETE", "FAILED", "CANCELLED"):
            break
        time.sleep(2)
    counts = job.counts if not isinstance(job.counts, list) else (job.counts[0] if job.counts else {})
    return {"job_id": str(job_id), "status": str(job.status), "counts": counts, "elapsed_s": time.time() - t0}


_CLIENT = None  # set in main


def main() -> int:
    global _CLIENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--p", type=int, default=1, help="QAOA depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots-log2", type=int, default=12, help="2^shots-log2 measurements per circuit")
    parser.add_argument("--skip-remote", action="store_true", help="Only run local emulator (no real chip)")
    parser.add_argument("--out", type=Path, default=Path("results/experiment_c_octillion.json"))
    args = parser.parse_args()

    # Read token from environment
    token = os.environ.get("BAUMAN_OCTILLION_TOKEN")
    if not token:
        raise SystemExit("BAUMAN_OCTILLION_TOKEN not set (source .env first)")

    from octillion.client import Client
    _CLIENT = Client(token)

    problem = make_problem(n=args.n, K=args.K, seed=args.seed)

    print(f"== Markowitz problem: N={args.n} K={args.K} seed={args.seed} ==")
    bf = brute_force_subsets(problem, args.K)
    print("Brute-force ranked subsets:")
    for r in bf:
        print(f"  subset={r['subset']} obj={r['objective']:.6f}")
    bf_best = bf[0]

    print("\n== Tabu reference ==")
    tabu = solve_with_tabu_selection(problem, num_reads=200, tenure=min(20, args.n-1), seed=args.seed)
    print(f"  subset={tabu.solver_meta['subset']} obj={tabu.objective:.6f}")

    pauli_coeffs, offset = qubo_to_ising(problem, args.K)
    print(f"\n== Ising Hamiltonian ({len(pauli_coeffs)} Pauli terms, offset={offset:.4f}) ==")
    for k, v in sorted(pauli_coeffs.items()):
        print(f"  {k}: {v:+.4f}")

    print(f"\n== Training QAOA(p={args.p}) on statevector simulator ==")
    params, ising_min = train_qaoa(pauli_coeffs, problem, args.K, p=args.p, seed=args.seed)
    print(f"  optimal params = {params}")
    print(f"  Ising expectation = {ising_min:.6f}, + offset = {ising_min + offset:.6f}")

    # Build the parameter-bound circuit
    bound = qaoa_circuit(pauli_coeffs, args.n, params, args.p)

    print(f"\n== Submitting to Octillion local emulator (Snowdrop 4q ver2) ==")
    local_back = _CLIENT.local("Snowdrop 4q ver2")
    cmap = [list(pair) for pair in local_back.coupling_map]
    local_transpiled = transpile(
        bound, basis_gates=list(local_back.basis_gates),
        coupling_map=cmap, optimization_level=1, seed_transpiler=args.seed,
    )
    print(f"  transpiled depth = {local_transpiled.depth()}, cz count = {local_transpiled.count_ops().get('cz', 0)}")
    local_res = run_on_octillion(local_transpiled, local_back, args.shots_log2, project="cif_qaoa_local")
    local_counts = local_res["counts"] if isinstance(local_res["counts"], dict) else (local_res["counts"][0] if local_res["counts"] else {})
    local_e, local_meta = expected_objective_from_counts(local_counts, problem, args.K)
    print(f"  E[obj | local emulator] = {local_e:.6f}  P(valid)={local_meta['p_valid']:.3f}")
    print(f"  best sampled subset: {local_meta['best_bitstring']} (obj={local_meta['best_obj']:.6f})")

    remote_payload = {"skipped": True}
    if not args.skip_remote:
        print(f"\n== Submitting to Octillion remote chip (Snowdrop 4q ver2) ==")
        remote_back = _CLIENT.remote("Snowdrop 4q ver2")
        rmap = [list(pair) for pair in remote_back.coupling_map]
        remote_transpiled = transpile(
            bound, basis_gates=list(remote_back.basis_gates),
            coupling_map=rmap, optimization_level=1, seed_transpiler=args.seed,
        )
        print(f"  transpiled depth = {remote_transpiled.depth()}, cz count = {remote_transpiled.count_ops().get('cz', 0)}")
        remote_res = run_on_octillion(remote_transpiled, remote_back, args.shots_log2,
                                       project=f"cif_qaoa_remote_{datetime.utcnow().strftime('%H%M%S')}")
        remote_counts = remote_res["counts"] if isinstance(remote_res["counts"], dict) else (remote_res["counts"][0] if remote_res["counts"] else {})
        remote_e, remote_meta = expected_objective_from_counts(remote_counts, problem, args.K)
        print(f"  E[obj | real chip] = {remote_e:.6f}  P(valid)={remote_meta['p_valid']:.3f}")
        print(f"  best sampled subset: {remote_meta['best_bitstring']} (obj={remote_meta['best_obj']:.6f})")
        remote_payload = {**remote_res, "expected_obj": remote_e, "meta": remote_meta}

    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "n": args.n, "K": args.K, "p": args.p, "seed": args.seed,
        "shots_log2": args.shots_log2,
        "brute_force": bf,
        "bf_best": bf_best,
        "tabu": {"subset": tabu.solver_meta["subset"], "objective": tabu.objective},
        "qaoa_params": params.tolist(),
        "qaoa_ising_min": ising_min,
        "ising_offset": offset,
        "pauli_terms": {k: v for k, v in pauli_coeffs.items()},
        "local_emulator": {**local_res, "expected_obj": local_e, "meta": local_meta},
        "remote_chip": remote_payload,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
