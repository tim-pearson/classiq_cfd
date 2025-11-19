import json
import random
import os
from classiq.applications.hamiltonian.pauli_decomposition import (
    hamiltonian_to_matrix,
    matrix_to_hamiltonian,
    matrix_to_pauli_operator,
)
from classiq import *
import numpy as np

def make_real_if_close(vec, tol=1e-8):
    """If imaginary parts are small compared to tol, return real part; otherwise
    return original.
"""
    if np.max(np.abs(np.imag(vec))) < tol:
        return np.real(vec)
    return vec


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def fidelity(u, v):
    """Fidelity between two normalized states (complex).
"""
    u = normalize(u)
    v = normalize(v)
    return np.abs(np.vdot(u, v)) ** 2















