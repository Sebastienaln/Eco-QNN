from math import gamma

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize


# -- Configuration -------------------------------------------------------------

# Dataset
N = 500  # Number of generated points
DIM = 4  # n-sphere dimension (2 reproduces the original 2D case)
TEST_SIZE = 0.2  # Fraction used for test set
SEED = 1  # Random seed (data split & optimizer init)

# Model
RC = 4  # Number of circuit layers

# Optimization
MAXITER = 300  # COBYLA max iterations

# Simulation
USE_NOISE = False  # True -> noisy AerSimulator | False -> exact Statevector
NOISE_RATE = 0.005  # Depolarizing error rate per gate (USE_NOISE=True only)
SHOTS = 1024  # Number of shots (USE_NOISE=True only)


def nsphere_radius_for_half_hypercube(dim):
    """Return radius r such that n-ball volume equals half of [-1,1]^n volume."""
    unit_ball_volume = np.pi ** (dim / 2) / gamma(dim / 2 + 1)
    target_volume = 2 ** (dim - 1)
    return (target_volume / unit_ball_volume) ** (1 / dim)


def reduce_to_3d(x):
    """Project an nD vector to 3 components for SU(2) parameterization."""
    x = np.asarray(x)
    if x.size == 1:
        return np.array([x[0], 0.0, 0.0])
    if x.size == 2:
        return np.array([x[0], x[1], 0.0])
    if x.size == 3:
        return x

    chunks = np.array_split(x, 3)
    return np.array([float(np.mean(c)) for c in chunks])


def generate_nsphere_dataset(num_points, dim, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(num_points, dim))
    r = nsphere_radius_for_half_hypercube(dim)
    labels = np.array([0 if np.sum(x**2) < r**2 else 1 for x in points])
    return points, labels, r


def train_test_split(X, y, test_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def U_su2(q, theta, omega, x, qubit):
    x3d = reduce_to_3d(x)
    val_0 = theta[0] + omega[0] * x3d[0]
    val_1 = theta[1] + omega[1] * x3d[1]
    val_2 = theta[2] + omega[2] * x3d[2]
    q.u(val_0, val_1, val_2, qubit)


def create_circuit(x, theta, omega):
    qc = QuantumCircuit(1)
    for i in range(RC):
        U_su2(qc, theta[i], omega[i], x, 0)
    return qc


def build_simulator():
    if USE_NOISE:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(NOISE_RATE, 1), ["u"])
        return AerSimulator(noise_model=nm)
    return StatevectorSimulator()


sv_sim = build_simulator()


def get_probs_batch(circuits):
    if USE_NOISE:
        meas = [c.copy() for c in circuits]
        for c in meas:
            c.measure_all()
        compiled = transpile(meas, sv_sim)
        results = sv_sim.run(compiled, shots=SHOTS).result()
        return [
            (
                results.get_counts(i).get("0", 0) / SHOTS,
                results.get_counts(i).get("1", 0) / SHOTS,
            )
            for i in range(len(meas))
        ]

    compiled = transpile(circuits, sv_sim)
    results = sv_sim.run(compiled).result()
    return [
        (
            float(np.abs(results.get_statevector(i).data[0]) ** 2),
            float(np.abs(results.get_statevector(i).data[1]) ** 2),
        )
        for i in range(len(circuits))
    ]


def cost_function_weighted(params, X_train, y_train):
    theta = params[: 3 * RC].reshape(RC, 3)
    omega = params[3 * RC : 6 * RC].reshape(RC, 3)
    alphas = params[6 * RC :]  # alpha_0, alpha_1

    circuits = [create_circuit(x, theta, omega) for x in X_train]
    probs = get_probs_batch(circuits)

    total_cost = 0.0
    for i, y_target in enumerate(y_train):
        f0, f1 = probs[i]
        y_expected = [1.0, 0.0] if y_target == 0 else [0.0, 1.0]
        total_cost += (alphas[0] * f0 - y_expected[0]) ** 2
        total_cost += (alphas[1] * f1 - y_expected[1]) ** 2

    return 0.5 * (total_cost / len(y_train))


def unpack_params(params):
    theta = params[: 3 * RC].reshape(RC, 3)
    omega = params[3 * RC : 6 * RC].reshape(RC, 3)
    alphas = params[6 * RC :]
    return theta, omega, alphas


def optimize_parameters(X_train, y_train):
    rng = np.random.default_rng(SEED)
    init = rng.uniform(-np.pi, np.pi, size=6 * RC + 2)

    cache = {"cost": None}
    cost_history = []
    step = {"k": 0}

    def objective(params):
        cost = cost_function_weighted(params, X_train, y_train)
        cache["cost"] = cost
        return cost

    def cb(_):
        step["k"] += 1
        cost_history.append(cache["cost"])
        print(f"Step {step['k']:03d} | cost = {cache['cost']:.6f}")

    res = minimize(
        objective,
        init,
        method="COBYLA",
        callback=cb,
        options={"maxiter": MAXITER},
    )

    theta_opt, omega_opt, _ = unpack_params(res.x)
    print(f"Final cost = {res.fun:.6f}")
    return theta_opt, omega_opt, res, cost_history


def predict_batch(X, theta, omega):
    circuits = [create_circuit(x, theta, omega) for x in X]
    probs = get_probs_batch(circuits)
    return np.array([0 if p0 >= 0.5 else 1 for p0, _ in probs])


def evaluate_metrics(X, y, theta, omega, positive_label=1):
    y_pred = predict_batch(X, theta, omega)
    tp = np.sum((y_pred == positive_label) & (y == positive_label))
    fp = np.sum((y_pred == positive_label) & (y != positive_label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = np.mean(y_pred == y)
    return precision, accuracy, y_pred


def plot_if_2d(X, y, y_pred, r):
    if X.shape[1] != 2:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        X[y_pred == 0][:, 0],
        X[y_pred == 0][:, 1],
        c="red",
        label="label 0",
        alpha=0.8,
    )
    plt.scatter(
        X[y_pred == 1][:, 0],
        X[y_pred == 1][:, 1],
        c="blue",
        label="label 1",
        alpha=0.8,
    )
    circle = plt.Circle(
        (0, 0), r, color="black", fill=False, linestyle="--", linewidth=2, label="decision sphere"
    )
    plt.gca().add_patch(circle)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Test set with predicted labels")
    plt.legend()
    plt.show()


def main():
    X, y, radius = generate_nsphere_dataset(N, DIM, SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, TEST_SIZE, SEED)

    print("Dimension n:", DIM)
    print("n-sphere radius:", radius)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)
    print("points in label 0:", np.sum(y == 0))
    print("points in label 1:", np.sum(y == 1))

    theta_opt, omega_opt, res, cost_history = optimize_parameters(X_train, y_train)
    precision_test, accuracy_test, y_pred = evaluate_metrics(X_test, y_test, theta_opt, omega_opt)

    print("Final train cost:", res.fun)
    print("Test precision:", precision_test)
    print("Test accuracy: ", accuracy_test)

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(1, len(cost_history) + 1), cost_history, color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost vs. optimization step")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axhline(
        res.fun, color="red", linestyle=":", linewidth=1.2, label=f"Final cost = {res.fun:.4f}"
    )
    ax1.legend()
    plt.tight_layout()
    plt.show()

    plot_if_2d(X_test, y_test, y_pred, radius)


if __name__ == "__main__":
    main()