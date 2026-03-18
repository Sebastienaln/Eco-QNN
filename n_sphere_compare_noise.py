import time
from qiskit_machine_learning.algorithms import VQC
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SPSA 
from qiskit_aer import AerSimulator
from sklearn.model_selection import train_test_split

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import SamplerV2 as Sampler


# Dataset
n          = 500   # Number of generated points
TEST_SIZE  = 0.2    # Fraction used for test set
SEED       = 1      # Random seed (data split & optimizer init)
N_DIM      = 3     # Features dimension

# Noise 
NOISE_RATE = 0.02

def generate_nsphere_data(n_samples, n_dim, radius=None):
    # Génération de points entre -1 et 1
    X = np.random.uniform(-1, 1, (n_samples, n_dim))    
    # Le rayon par défaut est choisi pour équilibrer les classes
    if radius is None:
        radius = np.sqrt(n_dim / 3) 
        
    # Calcul de la norme euclidienne au carré
    dist_sq = np.sum(X**2, axis=1)
    y = (dist_sq >= radius**2).astype(int)
    
    return X, y, radius

X, y, R = generate_nsphere_data(n_samples=n, n_dim=N_DIM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)

sampler_ideal = Sampler(AerSimulator())


nm = NoiseModel()
# Erreur sur les portes simples (1 qubit)
err1 = depolarizing_error(NOISE_RATE, 1)
nm.add_all_qubit_quantum_error(err1, ['u', 'h', 'ry', 'rz'])
# Erreur sur l'intrication (2 qubits) - on met souvent 5x plus
err2 = depolarizing_error(NOISE_RATE * 5, 2)
nm.add_all_qubit_quantum_error(err2, ['cx'])
    
# On crée le backend (ton ancien AerSimulator)
backend = AerSimulator(noise_model=nm, method="density_matrix")
# On l'enveloppe pour le rendre compatible VQC
sampler_noisy = Sampler(backend)


optimizers = {
    "SPSA": SPSA(maxiter=100),
    "COBYLA": COBYLA(maxiter=100)
}

modes = {
    "Idéal": sampler_ideal,
    "Bruité": sampler_noisy
}

feature_map = zz_feature_map(feature_dimension=N_DIM, reps=2)
ansatz = real_amplitudes(num_qubits=N_DIM, reps=2)

comparison_results = []
comparison_histories = {}

def callback(*args):
    if len(args) > 1:
        cost_history.append(args[1])

for opt_name, opt in optimizers.items():
    for mode_name, sampler in modes.items():
        label = f"{opt_name} ({mode_name})"
        print(f"Lancement : {label}...")
        
        cost_history = []
        vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=opt,
            callback=callback
        )
        
        start = time.time()
        vqc.fit(X_train, y_train)
        duration = time.time() - start
        
        comparison_histories[label] = cost_history.copy()
        acc = vqc.score(X_test, y_test)
        
        comparison_results.append({
            "Optimiseur": opt_name,
            "Mode": mode_name,
            "Temps (s)": duration,
            "Accuracy": acc
        })

