from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TwoBlockHamiltonian:
    name: str
    A: Array
    B: Array
    a_terms: tuple[Array, ...] = ()
    b_terms: tuple[Array, ...] = ()

    @property
    def hamiltonian(self) -> Array:
        return self.A + self.B

    @property
    def dimension(self) -> int:
        return self.A.shape[0]


def pauli_x() -> Array:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def pauli_y() -> Array:
    return np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)


def pauli_z() -> Array:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def identity() -> Array:
    return np.eye(2, dtype=complex)


def kron_all(operators: list[Array]) -> Array:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def one_body_term(single_site: Array, site: int, n_qubits: int) -> Array:
    ops = [identity() for _ in range(n_qubits)]
    ops[site] = single_site
    return kron_all(ops)


def two_body_term(left: Array, left_site: int, right: Array, right_site: int, n_qubits: int) -> Array:
    ops = [identity() for _ in range(n_qubits)]
    ops[left_site] = left
    ops[right_site] = right
    return kron_all(ops)


def tfim_hamiltonian(n_qubits: int, coupling_j: float, field_h: float, periodic: bool = False) -> TwoBlockHamiltonian:
    x = pauli_x()
    z = pauli_z()

    zz_sum = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    x_sum = np.zeros_like(zz_sum)
    zz_terms: list[Array] = []
    x_terms: list[Array] = []

    last_bond = n_qubits if periodic else n_qubits - 1
    for site in range(last_bond):
        right_site = (site + 1) % n_qubits
        term = two_body_term(z, site, z, right_site, n_qubits)
        zz_terms.append(term)
        zz_sum += term

    for site in range(n_qubits):
        term = one_body_term(x, site, n_qubits)
        x_terms.append(term)
        x_sum += term

    return TwoBlockHamiltonian(
        name=f"TFIM_{n_qubits}q_J{coupling_j}_h{field_h}",
        A=coupling_j * zz_sum,
        B=field_h * x_sum,
        a_terms=tuple(coupling_j * term for term in zz_terms),
        b_terms=tuple(field_h * term for term in x_terms),
    )


def driven_tfim_hamiltonian(
    n_qubits: int,
    coupling_j: float,
    field_h: float,
    amplitude: float,
    omega: float,
    time: float,
    periodic: bool = False,
) -> TwoBlockHamiltonian:
    effective_field = field_h + amplitude * np.sin(omega * time)
    return tfim_hamiltonian(n_qubits=n_qubits, coupling_j=coupling_j, field_h=effective_field, periodic=periodic)


def xxz_hamiltonian(n_qubits: int, coupling_xy: float, coupling_z: float, periodic: bool = False) -> TwoBlockHamiltonian:
    x = pauli_x()
    y = pauli_y()
    z = pauli_z()

    zz_sum = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    xy_sum = np.zeros_like(zz_sum)
    zz_terms: list[Array] = []
    xy_terms: list[Array] = []

    last_bond = n_qubits if periodic else n_qubits - 1
    for site in range(last_bond):
        right_site = (site + 1) % n_qubits
        zz_term = two_body_term(z, site, z, right_site, n_qubits)
        xx_term = two_body_term(x, site, x, right_site, n_qubits)
        yy_term = two_body_term(y, site, y, right_site, n_qubits)
        zz_terms.append(zz_term)
        xy_terms.extend((xx_term, yy_term))
        zz_sum += zz_term
        xy_sum += xx_term + yy_term

    return TwoBlockHamiltonian(
        name=f"XXZ_{n_qubits}q_Jxy{coupling_xy}_Jz{coupling_z}",
        A=coupling_z * zz_sum,
        B=coupling_xy * xy_sum,
        a_terms=tuple(coupling_z * term for term in zz_terms),
        b_terms=tuple(coupling_xy * term for term in xy_terms),
    )
