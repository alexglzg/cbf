"""
mvie.py – MVIE2D: Maximum-Volume Inscribed Ellipsoid (2-D)
===========================================================

Ported 1-to-1 from the MVIE2D class in firi_node_sdmn.cpp.

Solves:
    max  det(L)
    s.t. ||L^T a_i|| + a_i^T d ≤ b_i,  i = 1 … m
         L lower-triangular, L₁₁ > 0, L₂₂ > 0

Ellipsoid: { L x + d : ||x|| ≤ 1 }

Method: log-barrier interior-point with Newton steps.
State vector: x = [L11, L21, L22, d1, d2]  (5 variables).

Paper reference: Wang et al. (2025), Section V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ellipsoid:
    L: np.ndarray   # shape (2, 2) lower-triangular
    d: np.ndarray   # shape (2,)  centre

    @property
    def volume(self) -> float:
        """Area of the inscribed ellipse  = π · |L₁₁ · L₂₂|."""
        return math.pi * abs(self.L[0, 0] * self.L[1, 1])

    @property
    def center(self) -> np.ndarray:
        return self.d.copy()

    @property
    def shape_matrix(self) -> np.ndarray:
        """Σ = L L^T  (positive semi-definite)."""
        return self.L @ self.L.T


import math


class MVIE2D:
    """
    Maximum-Volume Inscribed Ellipsoid for a 2-D polytope {x : A x ≤ b}.

    Parameters
    ----------
    outer_iters : int
        Number of log-barrier outer iterations (default 20).
    inner_iters : int
        Newton centering steps per outer iteration (default 40).
    mu : float
        Barrier parameter growth factor (default 4.0).
    """

    def __init__(
        self,
        outer_iters: int = 20,
        inner_iters: int = 40,
        mu: float = 4.0,
    ):
        self._outer = outer_iters
        self._inner = inner_iters
        self._mu = mu

    # ------------------------------------------------------------------
    def solve(
        self,
        A: np.ndarray,        # shape (m, 2)
        b: np.ndarray,        # shape (m,)
        center_hint: np.ndarray,  # shape (2,)  — strictly interior point
    ) -> Ellipsoid:
        """
        Compute the MVIE of the polytope { x : A x ≤ b }.

        Returns an :class:`Ellipsoid`.  Falls back to a tiny ball if
        the polytope is degenerate.
        """
        m = A.shape[0]
        c = center_hint.copy().astype(float)

        # ── Initialisation: Chebyshev ball at center_hint ─────────────
        r = 1e18
        for i in range(m):
            norm_ai = np.linalg.norm(A[i])
            if norm_ai > 1e-10:
                gap = b[i] - A[i].dot(c)
                r = min(r, gap / norm_ai)

        if r <= 0 or not math.isfinite(r):
            r = 1e-4
        r *= 0.9    # strictly feasible
        r = max(r, 1e-6)

        # State: x = [L11, L21, L22, d1, d2]
        x = np.array([r, 0.0, r, c[0], c[1]])

        # ── Log-barrier method ────────────────────────────────────────
        t = 1.0
        for _ in range(self._outer):
            for _ in range(self._inner):
                g = self._gradient(A, b, x, t)
                H = self._hessian(A, b, x, t)
                H += 1e-8 * np.eye(5)

                try:
                    dx = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    break

                lambda_sq = -g.dot(dx)
                if lambda_sq < 1e-6:
                    break

                # Backtracking line search
                alpha = 1.0
                f0 = self._objective(A, b, x, t)
                for _ in range(32):
                    xn = x + alpha * dx
                    if xn[0] > 1e-10 and xn[2] > 1e-10:
                        fn = self._objective(A, b, xn, t)
                        if math.isfinite(fn) and fn < f0 + 0.3 * alpha * g.dot(dx):
                            x = xn
                            break
                    alpha *= 0.5
                    if alpha < 1e-12:
                        break

            if m / t < 1e-3:
                break
            t *= self._mu

        L = np.array([[x[0], 0.0],
                       [x[1], x[2]]])
        d = np.array([x[3], x[4]])
        return Ellipsoid(L=L, d=d)

    # ── Private helpers ───────────────────────────────────────────────

    def _objective(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        t: float,
    ) -> float:
        L11, L21, L22, d1, d2 = x
        if L11 <= 0 or L22 <= 0:
            return 1e18
        val = -t * (math.log(L11) + math.log(L22))
        for i in range(A.shape[0]):
            a1, a2 = A[i, 0], A[i, 1]
            r1 = L11 * a1 + L21 * a2
            r2 = L22 * a2
            gap = b[i] - a1 * d1 - a2 * d2 - math.sqrt(r1 * r1 + r2 * r2)
            if gap <= 0:
                return 1e18
            val -= math.log(gap)
        return val

    def _gradient(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        t: float,
    ) -> np.ndarray:
        L11, L21, L22, d1, d2 = x
        g = np.zeros(5)
        g[0] = -t / L11
        g[2] = -t / L22

        for i in range(A.shape[0]):
            a1, a2 = A[i, 0], A[i, 1]
            r1 = L11 * a1 + L21 * a2
            r2 = L22 * a2
            nr = math.sqrt(r1 * r1 + r2 * r2)
            gap = b[i] - a1 * d1 - a2 * d2 - nr
            gap = max(gap, 1e-15)
            ig = 1.0 / gap
            if nr > 1e-15:
                inr = 1.0 / nr
                g[0] += ig * r1 * a1 * inr   # ∂/∂L11
                g[1] += ig * r1 * a2 * inr   # ∂/∂L21
                g[2] += ig * r2 * a2 * inr   # ∂/∂L22
            g[3] += ig * a1                   # ∂/∂d1
            g[4] += ig * a2                   # ∂/∂d2
        return g

    def _hessian(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Numerical Hessian via central differences on the gradient."""
        eps = 1e-6
        H = np.zeros((5, 5))
        for j in range(5):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            H[:, j] = (self._gradient(A, b, xp, t)
                       - self._gradient(A, b, xm, t)) / (2.0 * eps)
        return 0.5 * (H + H.T)
