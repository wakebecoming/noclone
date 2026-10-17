"""Tests for noclone — analytic checks only, no hardware required."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noclone import (basis_cloner, bh_cloner, bh_images, fidelity, ket, plus,
                     purity, random_qubit, reduced_2q, reduced_3q,
                     run_basis_cloner, run_bh_cloner)


class TestBasisCloner(unittest.TestCase):
    def test_clones_zero_and_one_exactly(self):
        for b in (0, 1):
            out = run_basis_cloner(ket(b))
            ideal = np.kron(ket(b), ket(b))
            self.assertAlmostEqual(abs(np.vdot(ideal, out)), 1.0, places=12)

    def test_plus_gives_bell_state(self):
        out = run_basis_cloner(plus())
        bell = (ket(0, 0) + ket(1, 1)) / np.sqrt(2)
        self.assertTrue(np.allclose(out, bell) or np.allclose(out, -bell))

    def test_plus_clone_fidelity_is_exactly_half(self):
        out = run_basis_cloner(plus())
        ideal = np.kron(plus(), plus())
        f = abs(np.vdot(ideal, out)) ** 2
        self.assertAlmostEqual(f, 0.5, places=12)

    def test_plus_output_is_entangled_not_copied(self):
        out = run_basis_cloner(plus())
        rho_a = reduced_2q(out)
        # maximally mixed marginal -> Schmidt rank 2 -> entangled
        self.assertAlmostEqual(purity(rho_a), 0.5, places=12)

    def test_is_unitary(self):
        U = basis_cloner()
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(4)))


class TestOptimalCloner(unittest.TestCase):
    def test_images_are_orthonormal(self):
        c0, c1 = bh_images()
        self.assertAlmostEqual(np.linalg.norm(c0), 1.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(c1), 1.0, places=12)
        self.assertAlmostEqual(abs(np.vdot(c0, c1)), 0.0, places=12)

    def test_completed_map_is_unitary(self):
        U = bh_cloner()
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10))

    def test_fidelity_is_five_sixths_on_random_inputs(self):
        U = bh_cloner()
        rng = np.random.default_rng(1234)
        n = 120
        fa = fb = 0.0
        for _ in range(n):
            psi = random_qubit(rng)
            ra, rb = run_bh_cloner(psi, U)
            fa += fidelity(ra, psi)
            fb += fidelity(rb, psi)
        self.assertAlmostEqual(fa / n, 5 / 6, places=3)
        self.assertAlmostEqual(fb / n, 5 / 6, places=3)

    def test_copies_are_symmetric(self):
        U = bh_cloner()
        rng = np.random.default_rng(99)
        for _ in range(20):
            psi = random_qubit(rng)
            ra, rb = run_bh_cloner(psi, U)
            self.assertAlmostEqual(fidelity(ra, psi), fidelity(rb, psi), places=9)

    def test_copies_are_mixed_not_pure(self):
        # a perfect copy would leave pure marginals; 5/6 fidelity means mixed
        U = bh_cloner()
        psi = plus()
        ra, _ = run_bh_cloner(psi, U)
        self.assertLess(purity(ra), 1.0)
        self.assertGreater(purity(ra), 0.5)


if __name__ == "__main__":
    unittest.main()
