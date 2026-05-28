"""QAOA for cardinality-constrained Markowitz on Qiskit Aer (CPU or GPU).

The circuit uses the standard QAOA ansatz ``e^{-iβH_mix} e^{-iγH_cost}`` applied
to the binary-inclusion QUBO produced by
:func:`cif.qubo.builder.build_selection_bqm`. Classical optimisation of
``(β, γ)`` is driven by :func:`scipy.optimize.minimize` with the COBYLA
method. After the variational loop converges we sample the final state and
return the lowest-energy bit-string, refined via continuous mean-variance on
the chosen subset.

Two simulation backends are wired in:

* **CPU statevector** — default, works out of the box up to ~12–14 qubits
  on a laptop.
* **GPU statevector (cuQuantum)** — used on the HSE HPC cluster where NVIDIA
  GPUs are available. Scales up to ~24–28 qubits depending on GPU memory.
  Selected via ``device="GPU"`` which Qiskit Aer will route to
  ``AerSimulator(method="statevector", device="GPU")``. Requires
  ``pip install cuquantum-python`` and a CUDA-enabled Aer build (typically
  ``qiskit-aer-gpu`` on HPC).

The module is deliberately thin: the classical optimiser and shot handling
live in :func:`run_qaoa_selection`; individual circuit construction is in
:func:`_build_qaoa_circuit`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import dimod
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from cif.classical.continuous import solve_continuous_mvo
from cif.problem import PortfolioProblem, Solution
from cif.qubo.builder import build_selection_bqm, selection_bits_to_subset


@dataclass
class QaoaConfig:
    p: int = 2
    shots: int = 4096
    optimizer: str = "COBYLA"
    optimizer_maxiter: int = 100
    device: str = "CPU"  # "CPU" or "GPU"
    seed: int | None = 42
    init_params: np.ndarray | None = None


def _bqm_to_pauli(bqm: dimod.BinaryQuadraticModel) -> tuple[SparsePauliOp, float, list[int]]:
    """Convert a binary QUBO (variables in ``{0,1}``) into an Ising Pauli-Z operator.

    For binary ``x_i``, substitute ``x_i = (1 − z_i) / 2`` where ``z_i ∈ {−1, +1}``
    is the Pauli-Z eigenvalue. The resulting operator is diagonal in the
    computational basis and acts as ``H_C |bits⟩ = energy(bits) |bits⟩``.
    """
    variables = list(bqm.variables)
    n_qubits = len(variables)
    var_index = {v: i for i, v in enumerate(variables)}

    pauli_terms: dict[str, float] = {}
    offset = float(bqm.offset)

    for var, lin_coef in bqm.linear.items():
        q = var_index[var]
        # lin_coef * x = lin_coef * (1 - z)/2 = lin_coef/2 - lin_coef/2 * z
        offset += lin_coef / 2
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        pauli_terms["".join(label)] = pauli_terms.get("".join(label), 0.0) - lin_coef / 2

    for (u, v), quad_coef in bqm.quadratic.items():
        qu = var_index[u]
        qv = var_index[v]
        # x_u x_v = (1 - z_u)(1 - z_v)/4 = 1/4 - z_u/4 - z_v/4 + z_u z_v / 4
        offset += quad_coef / 4
        label_u = ["I"] * n_qubits
        label_u[n_qubits - 1 - qu] = "Z"
        pauli_terms["".join(label_u)] = pauli_terms.get("".join(label_u), 0.0) - quad_coef / 4
        label_v = ["I"] * n_qubits
        label_v[n_qubits - 1 - qv] = "Z"
        pauli_terms["".join(label_v)] = pauli_terms.get("".join(label_v), 0.0) - quad_coef / 4
        label_uv = ["I"] * n_qubits
        label_uv[n_qubits - 1 - qu] = "Z"
        label_uv[n_qubits - 1 - qv] = "Z"
        pauli_terms["".join(label_uv)] = pauli_terms.get("".join(label_uv), 0.0) + quad_coef / 4

    clean = {k: v for k, v in pauli_terms.items() if abs(v) > 1e-12}
    paulis = list(clean.keys())
    coeffs = np.asarray(list(clean.values()), dtype=float)
    op = SparsePauliOp(paulis, coeffs=coeffs)
    return op, offset, variables


def _build_qaoa_circuit(
    cost_op: SparsePauliOp,
    n_qubits: int,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    """Construct a QAOA circuit of depth ``len(gammas)``."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for gamma, beta in zip(gammas, betas):
        # Cost unitary: exp(-i γ H_C)
        for pauli, coef in zip(cost_op.paulis, cost_op.coeffs.real):
            z_positions = [
                n_qubits - 1 - i
                for i, p in enumerate(str(pauli))
                if p == "Z"
            ]
            if not z_positions:
                continue
            if len(z_positions) == 1:
                qc.rz(2 * gamma * coef, z_positions[0])
            elif len(z_positions) == 2:
                q0, q1 = z_positions
                qc.cx(q0, q1)
                qc.rz(2 * gamma * coef, q1)
                qc.cx(q0, q1)
            else:
                raise NotImplementedError("Only 1- and 2-local ZZ terms are expected for QUBOs")
        # Mixer: exp(-i β H_M), H_M = sum_i X_i
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    qc.measure_all()
    return qc


def _energy_from_counts(
    counts: dict[str, int],
    cost_op: SparsePauliOp,
    offset: float,
) -> tuple[float, dict[str, float]]:
    total = sum(counts.values())
    energies: dict[str, float] = {}
    expectation = 0.0
    for bitstring, count in counts.items():
        spins = np.array([1 - 2 * int(b) for b in bitstring])  # bit 0 -> +1, bit 1 -> -1
        e = offset
        for pauli, coef in zip(cost_op.paulis, cost_op.coeffs.real):
            prod = 1.0
            for i, p in enumerate(str(pauli)):
                if p == "Z":
                    prod *= spins[len(bitstring) - 1 - i]
            e += coef * prod
        energies[bitstring] = e
        expectation += e * count / total
    return expectation, energies


def run_qaoa_selection(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    config: QaoaConfig | None = None,
    refine_continuous: bool = True,
) -> Solution:
    """End-to-end QAOA pipeline for cardinality-constrained Markowitz.

    Same two-stage layout as :func:`cif.quantum.neal_sampler.solve_with_neal_selection`:
    QAOA picks the size-``K`` subset, cvxpy refines the continuous weights on
    that subset, and we return the resulting :class:`Solution`.
    """
    config = config or QaoaConfig()
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)

    bqm = build_selection_bqm(problem, cardinality=K)
    cost_op, offset, variables = _bqm_to_pauli(bqm)
    n_qubits = len(variables)

    simulator_kwargs = {"method": "statevector"}
    if config.device.upper() == "GPU":
        simulator_kwargs["device"] = "GPU"
    simulator = AerSimulator(**simulator_kwargs)

    rng = np.random.default_rng(config.seed)
    if config.init_params is not None:
        theta0 = np.asarray(config.init_params, dtype=float)
    else:
        theta0 = rng.uniform(0, np.pi, size=2 * config.p)

    def split(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[: config.p], theta[config.p :]

    def expectation(theta: np.ndarray) -> float:
        gammas, betas = split(theta)
        qc = _build_qaoa_circuit(cost_op, n_qubits, gammas, betas)
        transpiled = transpile(qc, simulator, optimization_level=1)
        result = simulator.run(transpiled, shots=config.shots, seed_simulator=config.seed).result()
        counts = result.get_counts()
        exp, _ = _energy_from_counts(counts, cost_op, offset)
        return exp

    t0 = time.perf_counter()
    opt_result = minimize(
        expectation,
        theta0,
        method=config.optimizer,
        options={"maxiter": config.optimizer_maxiter, "rhobeg": 0.3},
    )
    optimize_wall = time.perf_counter() - t0

    # Final sampling with optimal parameters
    gammas_best, betas_best = split(opt_result.x)
    qc_best = _build_qaoa_circuit(cost_op, n_qubits, gammas_best, betas_best)
    transpiled = transpile(qc_best, simulator, optimization_level=1)
    t_sample = time.perf_counter()
    result = simulator.run(transpiled, shots=config.shots, seed_simulator=config.seed).result()
    counts = result.get_counts()
    sample_wall = time.perf_counter() - t_sample
    _, energies = _energy_from_counts(counts, cost_op, offset)

    # Refine top feasible subsets
    ranked = sorted(energies.items(), key=lambda kv: kv[1])
    best: Solution | None = None
    evaluated = 0
    refine_t0 = time.perf_counter()
    for bitstring, _energy in ranked:
        bit_array = np.array([int(b) for b in bitstring])[::-1]  # qiskit little-endian
        sample_dict = {variables[i]: int(bit_array[i]) for i in range(n_qubits)}
        subset = selection_bits_to_subset(sample_dict, problem.n)
        if len(subset) != K:
            continue
        if refine_continuous:
            idx = np.asarray(subset)
            sub_problem = PortfolioProblem(
                mu=problem.mu[idx],
                sigma=problem.sigma[np.ix_(idx, idx)],
                asset_names=tuple(problem.asset_names[i] for i in subset),
                w_min=problem.w_min,
                w_max=problem.w_max,
                budget=problem.budget,
                risk_aversion=problem.risk_aversion,
            )
            try:
                sub_sol = solve_continuous_mvo(sub_problem)
            except RuntimeError:
                continue
            weights = np.zeros(problem.n, dtype=float)
            weights[idx] = sub_sol.weights
        else:
            weights = np.zeros(problem.n, dtype=float)
            weights[list(subset)] = 1.0 / K
        obj = problem.objective_value(weights)
        evaluated += 1
        if best is None or obj < best.objective:
            best = Solution(
                weights=weights,
                objective=obj,
                feasible=True,
                wall_time_s=0.0,
                solver=f"qaoa_selection/p={config.p}/{config.device}",
                solver_meta={
                    "subset": list(subset),
                },
            )
        if evaluated >= 20:
            break
    refine_wall = time.perf_counter() - refine_t0

    if best is None:
        raise RuntimeError("QAOA did not find any size-K subset among top samples")

    total_wall = optimize_wall + sample_wall + refine_wall
    best.wall_time_s = total_wall
    best.solver_meta.update({
        "p": config.p,
        "shots": config.shots,
        "device": config.device,
        "optimizer": config.optimizer,
        "optimizer_iters": int(getattr(opt_result, "nit", getattr(opt_result, "nfev", -1))),
        "optimizer_success": bool(opt_result.success),
        "optimize_wall_s": optimize_wall,
        "sample_wall_s": sample_wall,
        "refine_wall_s": refine_wall,
        "final_theta": list(map(float, opt_result.x)),
        "n_qubits": n_qubits,
        "n_evaluated": evaluated,
    })
    return best
