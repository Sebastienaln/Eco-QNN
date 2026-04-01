#!/usr/bin/env python3
"""
Batch script for running the re-uploading study multiple times across different seeds.
Designed for supercomputer execution with output to Excel.

Usage:
  python run_reupload_study.py --repeats 10 --n-samples 1000 --maxiter 200 --seed 1
  python run_reupload_study.py --help
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize


def generate_nsphere_data(n_samples: int, n_dim: int, radius=None, seed=None):
    """Generate random points in [-1,1]^n_dim sphere classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_dim))
    r = radius if radius is not None else np.sqrt(n_dim / 3)
    y = (np.sum(X ** 2, axis=1) >= r ** 2).astype(int)
    return X, y, r


def build_backend(use_noise: bool, noise_rate: float):
    """Build simulator backend with optional noise model."""
    if use_noise:
        nm = NoiseModel()
        err1 = depolarizing_error(noise_rate, 1)
        nm.add_all_qubit_quantum_error(err1, ["u", "h", "ry", "rz"])
        err2 = depolarizing_error(noise_rate * 5, 2)
        nm.add_all_qubit_quantum_error(err2, ["cx"])
        return AerSimulator(noise_model=nm, method="density_matrix")
    return StatevectorSimulator()


def create_reupload_circuit(x, theta, omega, num_layers, n_dim, measure=False):
    """Create re-uploading circuit for 1 qubit."""
    qc = QuantumCircuit(1)
    n_gates_per_layer = int(np.ceil(n_dim / 3))
    for r in range(num_layers):
        qc.h(0)
        for g in range(n_gates_per_layer):
            base = 3 * g
            params = []
            for i in range(3):
                idx = base + i
                val = theta[r, idx]
                if idx < n_dim:
                    val += omega[r, idx] * x[idx]
                params.append(val)
            qc.u(params[0], params[1], params[2], 0)
    if measure:
        qc.measure_all()
    return qc


def get_reupload_probs_batch(circuits, backend, use_noise: bool, shots: int):
    """Get measurement probabilities for circuit batch."""
    probs = []
    if use_noise:
        measured = []
        for qc in circuits:
            qc_m = qc.copy()
            if qc_m.num_clbits == 0:
                qc_m.measure_all()
            measured.append(qc_m)
        compiled = transpile(measured, backend)
        result = backend.run(compiled, shots=shots).result()
        for qc_t in compiled:
            counts = result.get_counts(qc_t)
            p0 = counts.get('0', 0) / shots
            p1 = counts.get('1', 0) / shots
            probs.append((p0, p1))
    else:
        compiled = transpile(circuits, backend)
        result = backend.run(compiled).result()
        for i in range(len(circuits)):
            sv = result.get_statevector(i)
            p0 = float(np.abs(sv[0]) ** 2)
            p1 = float(np.abs(sv[1]) ** 2)
            probs.append((p0, p1))
    return probs


def unpack_params(params, num_layers, n_dim):
    """Unpack flattened parameter vector into theta, omega, alphas."""
    n_gates_per_layer = int(np.ceil(n_dim / 3))
    n_params_per_layer = n_gates_per_layer * 3
    theta = params[:n_params_per_layer * num_layers].reshape(num_layers, n_params_per_layer)
    omega = params[n_params_per_layer * num_layers:2 * n_params_per_layer * num_layers].reshape(num_layers, n_params_per_layer)
    alphas = params[2 * n_params_per_layer * num_layers:]
    return theta, omega, alphas


def reupload_cost_weighted(params, X, y, num_layers, n_dim, backend, use_noise, shots):
    """Cost function for re-uploading circuit optimization."""
    theta, omega, alphas = unpack_params(params, num_layers, n_dim)
    circuits = [create_reupload_circuit(x, theta, omega, num_layers, n_dim) for x in X]
    probs = get_reupload_probs_batch(circuits, backend, use_noise, shots)
    total_cost = 0.0
    for i, y_target in enumerate(y):
        p0, p1 = probs[i]
        y_expected = (1.0, 0.0) if y_target == 0 else (0.0, 1.0)
        weight = alphas[y_target] ** 2
        total_cost += weight * ((p0 - y_expected[0]) ** 2 + (p1 - y_expected[1]) ** 2)
    return 0.5 * total_cost / len(X)


def optimize_reupload_parameters(X, y, num_layers, n_dim, maxiter, seed, backend, use_noise, shots):
    """Optimize re-uploading parameters using COBYLA."""
    n_gates_per_layer = int(np.ceil(n_dim / 3))
    n_params_per_layer = n_gates_per_layer * 3
    rng = np.random.default_rng(seed)
    init = rng.uniform(-np.pi, np.pi, size=2 * n_params_per_layer * num_layers + 2)

    def objective(params):
        return reupload_cost_weighted(params, X, y, num_layers, n_dim, backend, use_noise, shots)

    res = minimize(objective, init, method="COBYLA", options={"maxiter": maxiter})
    theta_opt, omega_opt, alphas_opt = unpack_params(res.x, num_layers, n_dim)
    return res, theta_opt, omega_opt, alphas_opt


def predict_reupload_batch(X, theta, omega, num_layers, n_dim, backend, use_noise, shots):
    """Predict labels for batch of inputs."""
    circuits = [create_reupload_circuit(x, theta, omega, num_layers, n_dim) for x in X]
    probs = get_reupload_probs_batch(circuits, backend, use_noise, shots)
    return np.array([0 if p0 >= p1 else 1 for p0, p1 in probs])


def evaluate_reupload(X, y, theta, omega, num_layers, n_dim, backend, use_noise, shots):
    """Evaluate accuracy on dataset."""
    y_pred = predict_reupload_batch(X, theta, omega, num_layers, n_dim, backend, use_noise, shots)
    accuracy = np.mean(y_pred == y)
    return {"accuracy": accuracy, "y_pred": y_pred}


def run_single_repeat(args_tuple):
    """Run a single repeat (used by multiprocessing.Pool)."""
    rep, seed, dimensions, layers_list, n_samples, test_size, maxiter, use_noise, noise_rate, shots = args_tuple
    records = []
    backend = build_backend(use_noise, noise_rate)
    
    print(f"Starting repeat {rep+1} (seed={seed})")
    
    for d in dimensions:
        X, y, _ = generate_nsphere_data(n_samples=n_samples, n_dim=d, radius=None, seed=seed)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(X))
        split_idx = int(len(X) * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for L in layers_list:
            res, t_opt, o_opt, a_opt = optimize_reupload_parameters(
                X_train, y_train, num_layers=L, n_dim=d, maxiter=maxiter, 
                seed=seed, backend=backend, use_noise=use_noise, shots=shots
            )
            metrics = evaluate_reupload(
                X_test, y_test, t_opt, o_opt, num_layers=L, n_dim=d, 
                backend=backend, use_noise=use_noise, shots=shots
            )
            records.append({
                "repeat": rep,
                "seed": seed,
                "dimension": d,
                "layers": L,
                "accuracy": metrics["accuracy"],
            })
    
    print(f"Completed repeat {rep+1}")
    return records


def run_study(dimensions, layers_list, n_samples, test_size, maxiter, use_noise, noise_rate, shots, base_seed, repeats, output_path: Path, n_jobs: int = 1):
    """Run full study across dimensions/layers with multiple repeats (seeds), parallelized."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel execution
    task_args = [
        (rep, base_seed + rep, dimensions, layers_list, n_samples, test_size, maxiter, use_noise, noise_rate, shots)
        for rep in range(repeats)
    ]
    
    print(f"Running {repeats} repeats with {n_jobs} parallel workers...")
    
    # Run in parallel
    all_records = []
    if n_jobs == 1:
        # Sequential execution
        for args in task_args:
            records = run_single_repeat(args)
            all_records.extend(records)
    else:
        # Parallel execution
        with Pool(n_jobs) as pool:
            results = pool.map(run_single_repeat, task_args)
            for records in results:
                all_records.extend(records)
    
    # Save to Excel
    df_raw = pd.DataFrame(all_records)
    summary = df_raw.groupby(["dimension", "layers"]).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        n_runs=("accuracy", "size")
    ).reset_index()
    
    with pd.ExcelWriter(output_path) as writer:
        df_raw.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    
    print(f"\nSaved results to {output_path}")
    print("Summary statistics:")
    print(summary.to_string())


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch runner for re-uploading circuit study (supercomputer-friendly with parallelization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 4, 5, 6], 
                   help="List of feature dimensions to test")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 14, 20],
                   help="List of circuit layer counts to test")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of samples per dimension per repeat")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test set fraction (0.0-1.0)")
    p.add_argument("--maxiter", type=int, default=200,
                   help="COBYLA max iterations per optimization")
    p.add_argument("--use-noise", action="store_true",
                   help="Enable depolarizing noise in simulator")
    p.add_argument("--noise-rate", type=float, default=0.02,
                   help="Depolarizing error rate (only if --use-noise)")
    p.add_argument("--shots", type=int, default=1024,
                   help="Measurement shots for noisy simulation")
    p.add_argument("--seed", type=int, default=1,
                   help="Base random seed (repeats use seed, seed+1, ...)")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of independent runs with different seeds")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Number of parallel workers (1=sequential, -1=all cores)")
    p.add_argument("--output", type=Path, default=Path("RESULTS/reupload_study_batch.xlsx"),
                   help="Output Excel file path")
    return p.parse_args()


def main():
    import os
    args = parse_args()
    
    # Handle n_jobs: -1 means all available cores
    n_cores_available = os.cpu_count()
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = n_cores_available
    
    print(f"Available cores: {n_cores_available}")
    print(f"Using {n_jobs} parallel worker(s)")
    
    run_study(
        dimensions=args.dimensions,
        layers_list=args.layers,
        n_samples=args.n_samples,
        test_size=args.test_size,
        maxiter=args.maxiter,
        use_noise=args.use_noise,
        noise_rate=args.noise_rate,
        shots=args.shots,
        base_seed=args.seed,
        repeats=args.repeats,
        output_path=args.output,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
