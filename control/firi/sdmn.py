"""
sdmn.py – SDMN2D: Small-Dimensional Minimum-Norm QP (2-D specialisation)
=========================================================================

Ported 1-to-1 from the SDMN2D class in firi_node_sdmn.cpp.

Solves:
    min  ½ ||y||²
    s.t. e_i^T y <= f_i,   i = 1 … d

Algorithm: Seidel's randomised incremental LP, lifted to minimum-norm QP.
Expected complexity O(d) for n=2 (linear in constraints).

Paper reference: Wang et al. (2025), Section IV, Algorithm 2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SDMNResult:
    y: np.ndarray   # shape (2,) — solution in 2-D
    feasible: bool


class SDMN2D:
    """
    2-D minimum-norm QP via Seidel's randomised incremental algorithm.

    Parameters
    ----------
    seed : int
        RNG seed for the random constraint permutation.  Use a fixed value
        for reproducibility; ``None`` picks a random seed.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def solve(
        self,
        e: List[np.ndarray],   # constraint normals,  each shape (2,)
        f: List[float],        # constraint offsets
    ) -> SDMNResult:
        """
        Solve  min ½||y||²  s.t.  e_i^T y ≤ f_i.

        Returns
        -------
        SDMNResult
            .y         – optimal 2-D point
            .feasible  – False if the QP is infeasible
        """
        d = len(e)
        if d == 0:
            return SDMNResult(y=np.zeros(2), feasible=True)

        # Random permutation [Paper §IV-B3: expected linear time]
        perm = list(range(d))
        self._rng.shuffle(perm)

        y = np.zeros(2)   # unconstrained minimum

        for ii in range(d):
            idx = perm[ii]

            # ── Violation check [Fig. 3(a)→(b): not violated, keep y] ──
            if e[idx].dot(y) <= f[idx] + 1e-12:
                continue

            # ── Violated → constraint active at optimum ──────────────
            # [Fig. 3(a)→(c): project to 1-D subproblem]
            eh = e[idx]
            fh = f[idx]
            eTe = eh.dot(eh)
            if eTe < 1e-15:
                return SDMNResult(y=np.zeros(2), feasible=False)

            # Minimum-norm point on the constraint plane [Paper Eq. 18]
            v = (fh / eTe) * eh

            # ── Householder reflection → 1-D basis [Paper Eq. 24-26] ─
            j = 0 if abs(v[0]) >= abs(v[1]) else 1
            k = 1 - j

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-15:
                m_col = np.array([-eh[1], eh[0]])
                mn = np.linalg.norm(m_col)
                if mn > 1e-15:
                    m_col /= mn
                else:
                    return SDMNResult(y=np.zeros(2), feasible=False)
            else:
                sign_vj = 1.0 if v[j] >= 0 else -1.0
                u_ref = v.copy()
                u_ref[j] += sign_vj * v_norm            # Householder vector
                uTu = u_ref.dot(u_ref)
                if uTu < 1e-15:
                    perp = np.array([-eh[1], eh[0]])
                    m_col = perp / (np.linalg.norm(perp) + 1e-30)
                else:
                    # Column k of H^T, i.e. M[:,k] [Paper: M = H^T without col j]
                    ek = np.zeros(2); ek[k] = 1.0
                    m_col = ek - (2.0 * u_ref[k] / uTu) * u_ref

            # ── 1-D sub-problem: min t² s.t. a_i t ≤ b_i ────────────
            # From previous constraints: e_p^T(m*t + v) ≤ f_p
            #   → (e_p^T m) * t ≤ f_p - e_p^T v        [Paper Eq. 23]
            lo, hi = -1e18, 1e18
            feasible = True

            for pp in range(ii):
                pidx = perm[pp]
                a1d = e[pidx].dot(m_col)
                b1d = f[pidx] - e[pidx].dot(v)

                if abs(a1d) < 1e-15:
                    if b1d < -1e-10:
                        feasible = False
                        break
                    continue
                bound = b1d / a1d
                if a1d > 0:
                    hi = min(hi, bound)
                else:
                    lo = max(lo, bound)

            if not feasible or lo > hi + 1e-10:
                return SDMNResult(y=np.zeros(2), feasible=False)

            # 1-D minimum-norm: closest to 0 in [lo, hi] [Paper Alg.2 Line 4]
            if lo <= 0.0 <= hi:
                t = 0.0
            elif lo > 0.0:
                t = lo
            else:
                t = hi

            y = m_col * t + v   # [Paper Eq. 21]

        return SDMNResult(y=y, feasible=True)
