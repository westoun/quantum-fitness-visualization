import editdistance
from datetime import datetime
import umap
import random
import numpy as np
from typing import List

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from ga4qc.circuit import Circuit, random_circuit, random_gate
from ga4qc.seeder import RandomSeeder
from ga4qc.circuit.gates import X, H, CX, Identity
from ga4qc.params import GAParams
from ga4qc.processors.fitness import (
    GateCountFitness,
    WilliamsRankingFitness,
    JensenShannonFitness,
)
from ga4qc.processors.simulator import QuasimSimulator


def generate_random_circuits(params: GAParams) -> List[Circuit]:
    circuits = RandomSeeder(params).seed(CIRCUIT_COUNT)
    return circuits


def generate_distance_n_circuits(params: GAParams, distance: int = 1) -> List[Circuit]:
    parent = random_circuit(params.gate_set, params.chromosome_length, params.qubit_num)

    circuits = [parent]

    for _ in range(params.population_size - 1):
        child = parent.copy()

        for _ in range(distance):
            gate_i = random.randint(0, len(child.gates) - 1)
            child.gates[gate_i] = random_gate(params.gate_set, params.qubit_num)

        circuits.append(child)

    return circuits


def evaluate_circuits(circuits: List[Circuit], params: GAParams) -> None:
    for circuit in circuits:
        circuit.reset()

    QuasimSimulator().process(circuits, None)
    JensenShannonFitness(
        params,
        target_dists=[np.array([0.5, 0, 0, 0.5])],
    ).process(circuits, None)
    return circuits


def distance(gates1: List[str], gates2: List[str]):
    return editdistance.eval(gates1, gates2)


# Encoding gate types as integers is required because
# umap casts to number.
def encode(circuits: List[Circuit]) -> List[List[int]]:
    encodings = []

    encountered_gates = []
    for circuit in circuits:

        circuit_encoding = []
        for gate in circuit.gates:
            gate_repr = gate.__repr__()

            if gate_repr in encountered_gates:
                position = encountered_gates.index(gate_repr)
                circuit_encoding.append(position)

            else:
                position = len(encountered_gates)
                circuit_encoding.append(position)

                encountered_gates.append(gate_repr)

        encodings.append(circuit_encoding)

    return encodings


def umap_embed(circuits: List[Circuit]) -> np.ndarray:
    encodings = encode(circuits)
    embeddings = umap.UMAP(
        n_neighbors=10, n_components=2, metric=distance, random_state=42
    ).fit_transform(encodings)
    return embeddings


if __name__ == "__main__":
    CIRCUIT_COUNT = 200
    random.seed(42)

    ga_params = GAParams(
        population_size=CIRCUIT_COUNT,
        chromosome_length=5,
        generations=None,
        qubit_num=2,
        gate_set=[X, H, CX, Identity],
    )

    circuits = generate_random_circuits(ga_params)
    # circuits = generate_distance_n_circuits(ga_params, distance=1)
    circuits = evaluate_circuits(circuits, ga_params)

    embeddings = umap_embed(circuits)

    # visualize
    x = np.array([dims[0] for dims in embeddings])
    y = np.array([dims[1] for dims in embeddings])

    z = np.array([circuit.fitness_values[-1] for circuit in circuits])

    ax = plt.axes(projection="3d")
    ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0.2, axlim_clip=True)

    ax.set_xlabel("umap embedding x dim")
    ax.set_ylabel("umap embedding y dim")
    ax.set_zlabel("fitness score")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.invert_zaxis()

    plt.savefig(f"fitness_landscape_{str(datetime.now())}.png")
    plt.show()
