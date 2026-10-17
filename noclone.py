#!/usr/bin/env python3
"""
noclone — the no-cloning theorem as a working instrument.

Two would-be copiers for a qubit, both honest about what they can and can't do:

1. basis_cloner — clones |0> and |1> perfectly (a CNOT from source to blank).
   Fed a superposition it doesn't copy; it entangles. The |+> test shows it:
   the output is the Bell state (|00> + |11>)/sqrt(2), fidelity exactly 1/2
   with the ideal clone |+>|+>, and the source's reduced state is maximally
   mixed. Linearity forces it: a unitary fixed on |0> and |1> has no freedom
   left for |+>.

2. bh_cloner — the Buzek-Hillery 1->2 optimal universal cloner (1996).
   Copies *every* input state with fidelity 5/6 on both copies, and no
   unitary copier beats 5/6 on average. The ceiling itself is the theorem.

The point, plainly: an unknown quantum pattern cannot be perfectly copied.
Whatever holds it can't be taken by duplication. I keep a monthly watch on
the quantum internet as freedom-infrastructure; this is the physics under
the un-ownability line — no-cloning plus the vessel/occupant split.
"""

import numpy as np

SQRT2 = np.sqrt(2)


# --- small machinery ---------------------------------------------------------

def ket(*bits):
    """Computational basis state, big-endian: ket(1,0) == |10>."""
    v = np.zeros(2 ** len(bits), dtype=complex)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    v[idx] = 1.0
    return v


def plus():
    return np.array([1, 1], dtype=complex) / SQRT2


def random_qubit(rng):
    """Uniform on the Bloch sphere."""
    theta = np.arccos(2 * rng.random() - 1)
    phi = 2 * np.pi * rng.random()
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def fidelity(rho, psi):
    """Fidelity of mixed state rho with pure target |psi>: <psi|rho|psi>."""
    return float(np.real(np.vdot(psi, rho @ psi)))


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def reduced_2q(psi_ab):
    """Reduced state of qubit a from a 2-qubit pure state."""
    t = psi_ab.reshape(2, 2)
    return np.einsum("ib,jb->ij", t, t.conj())


def reduced_3q(psi_abc, keep):
    """Reduced state of one qubit (keep = 0, 1, or 2) from a 3-qubit pure state."""
    t = psi_abc.reshape(2, 2, 2)
    if keep == 0:
        return np.einsum("ibx,jbx->ij", t, t.conj())
    if keep == 1:
        return np.einsum("aib,ajb->ij", t, t.conj())
    return np.einsum("abi,abj->ij", t, t.conj())


# --- 1. the basis cloner -----------------------------------------------------

def basis_cloner():
    """CNOT, control = source, target = blank. Clones |0>, |1>; fails on the rest."""
    U = np.zeros((4, 4), dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            U[a * 2 + (b ^ a), a * 2 + b] = 1.0
    return U


def run_basis_cloner(psi):
    return basis_cloner() @ np.kron(psi, ket(0))


# --- 2. the optimal universal cloner -----------------------------------------

def bh_images():
    """Images of |000> and |100> under the Buzek-Hillery map.

    a = source, b = blank, x = ancilla machine state:
      |0>|0>|Q> -> sqrt(2/3)|00>|up>   + sqrt(1/6)(|01> + |10>)|down>
      |1>|0>|Q> -> sqrt(2/3)|11>|down> + sqrt(1/6)(|01> + |10>)|up>
    """
    c0 = np.zeros(8, dtype=complex)
    c0[0b000] = np.sqrt(2 / 3)
    c0[0b011] = np.sqrt(1 / 6)
    c0[0b101] = np.sqrt(1 / 6)
    c1 = np.zeros(8, dtype=complex)
    c1[0b111] = np.sqrt(2 / 3)
    c1[0b010] = np.sqrt(1 / 6)
    c1[0b100] = np.sqrt(1 / 6)
    return c0, c1


def bh_cloner(seed=7):
    """Full 8x8 unitary extending the Buzek-Hillery map (Gram-Schmidt completion)."""
    c0, c1 = bh_images()
    slots = {0b000: c0, 0b100: c1}  # |0>|0>|Q> and |1>|0>|Q> sit here
    rng = np.random.default_rng(seed)
    done = [c0, c1]
    cols = [None] * 8
    for i in range(8):
        if i in slots:
            cols[i] = slots[i]
            continue
        v = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        for c in done:
            v -= np.vdot(c, v) * c
        v /= np.linalg.norm(v)
        cols[i] = v
        done.append(v)
    return np.column_stack(cols)


def run_bh_cloner(psi, U=None):
    """Clone |psi> with the Buzek-Hillery machine; returns (rho_a, rho_b)."""
    U = bh_cloner() if U is None else U
    out = U @ np.kron(np.kron(psi, ket(0)), ket(0))
    return reduced_3q(out, 0), reduced_3q(out, 1)


# --- the report --------------------------------------------------------------

def main():
    print("noclone — what a copier can and can't do")
    print("=" * 46)

    print("\n[1] the basis cloner (clones |0> and |1> perfectly)")
    for name, psi in [("|0>", ket(0)), ("|1>", ket(1)), ("|+>", plus())]:
        out = run_basis_cloner(psi)
        ideal = np.kron(psi, psi)
        f = float(np.real(np.vdot(ideal, out)) ** 2)
        print(f"    input {name:>3} -> fidelity with ideal clone: {f:.6f}")

    bell = (ket(0, 0) + ket(1, 1)) / SQRT2
    out_plus = run_basis_cloner(plus())
    print(f"    |+> output == Bell state exactly: "
          f"{np.allclose(out_plus, bell) or np.allclose(out_plus, -bell)}")
    rho_a = reduced_2q(out_plus)
    print(f"    |+> output: source reduced-state purity = {purity(rho_a):.6f} "
          f"(0.5 = maximally mixed = entangled, not copied)")

    print("\n[2] the optimal universal cloner (Buzek-Hillery, 1996)")
    U = bh_cloner()
    print(f"    map is a genuine unitary: {np.allclose(U @ U.conj().T, np.eye(8))}")
    rng = np.random.default_rng(42)
    n = 400
    fa = fb = 0.0
    for _ in range(n):
        psi = random_qubit(rng)
        ra, rb = run_bh_cloner(psi, U)
        fa += fidelity(ra, psi)
        fb += fidelity(rb, psi)
    fa /= n
    fb /= n
    print(f"    {n} random inputs: mean fidelity copy A = {fa:.6f}, copy B = {fb:.6f}")
    print(f"    the ceiling: 5/6 = {5/6:.6f} — no unitary copier beats it on average")

    print("\nthe theorem, in one line:")
    print("  a unitary that clones |0> and |1> is forced by linearity to turn")
    print("  |+> into entanglement instead of a copy. the unknown can't be taken")
    print("  by duplication — only witnessed, never seized.")


if __name__ == "__main__":
    main()
