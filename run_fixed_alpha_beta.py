#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim-Fusion (Fixed α,β) ― Reproduce the grid experiment in our paper
==================================================================

* Four data-generating scenarios (GMM / Gaussian / Uniform / Gamma)
* α, β  ∈ {0, 0.2, 0.6, 0.8, 1.0}  —— **treated as GIVEN (not estimated)**
* For each pair → generate model / obs sides, compute TL, EL and fused MSE
* Results are saved to   sim_fixed_alpha_beta.csv
------------------------------------------------------------------
This script is for **simulation study ONLY** (Section III-B of the paper),
where α and β are pre-specified grid values.
"""

import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 1. True data generators  (exactly as in Section III-A)
# ------------------------------------------------------------------
def gen_true_gmm(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    comps = rng.choice([0, 1], p=[0.4, 0.6], size=n)
    x = np.empty(n)
    x[comps == 0] = rng.normal(0.0, 1.0, (comps == 0).sum())
    x[comps == 1] = rng.normal(5.0, 2.0, (comps == 1).sum())
    return x


def gen_true_gauss(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(2.0, 3.0, n)


def gen_true_uniform(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 10.0, n)


def gen_true_gamma(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 2.0, n)


TRUE_GENS: Dict[int, Tuple[str, Callable[[int, int], np.ndarray]]] = {
    1: ("GMM 40:60", gen_true_gmm),
    2: ("Gaussian", gen_true_gauss),
    3: ("Uniform", gen_true_uniform),
    4: ("Gamma", gen_true_gamma),
}

# ------------------------------------------------------------------
# 2. Perturbations (model / obs)
# ------------------------------------------------------------------
def gen_model(data: np.ndarray, alpha: float, seed: int,
              offset_base: float = 0.3, noise_base: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return data + offset_base * alpha + rng.normal(0, noise_base * (1 - alpha), len(data))


def gen_obs(data: np.ndarray, beta: float, seed: int,
            offset_base: float = 0.2, noise_base: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return data - offset_base * beta + rng.normal(0, noise_base * (1 - beta), len(data))

# ------------------------------------------------------------------
# 3. Likelihood helpers
# ------------------------------------------------------------------
def tl_ll_mu(x: np.ndarray):
    mu, sig = x.mean(), x.std(ddof=1)
    if sig < 1e-14:
        return -1e15, mu
    ll = -0.5 * len(x) * math.log(2 * math.pi * sig * sig) - ((x - mu) ** 2).sum() / (
        2 * sig * sig
    )
    return ll, mu


def el_ll_mu(x: np.ndarray, max_iter: int = 200):
    n, ybar = len(x), x.mean()

    def g(lmb):
        d = 1 + lmb * (x - ybar)
        if np.any(d <= 0):
            return np.inf
        return (1 / d).sum() - n

    if abs(g(0)) < 1e-8:
        return -n * math.log(n), ybar

    a, b = -1e5, 1e5
    if g(a) * g(b) > 0:
        return -1e15, ybar
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        gm = g(m)
        if abs(gm) < 1e-8:
            break
        if g(a) * gm < 0:
            b = m
        else:
            a = m
    lam = 0.5 * (a + b)
    d = 1 + lam * (x - ybar)
    if np.any(d <= 0):
        return -1e15, ybar
    ll = (-np.log(n * d)).sum()
    return ll, ybar


def mse(y: np.ndarray, pred: float) -> float:
    return ((y - pred) ** 2).mean()

# ------------------------------------------------------------------
# 4. Core loop
# ------------------------------------------------------------------
def run_scenario(dist_id: int, n: int = 500) -> pd.DataFrame:
    name, gen_fun = TRUE_GENS[dist_id]
    y_true = gen_fun(n, seed=100 + dist_id)
    grid = [0, 0.2, 0.6, 0.8, 1.0]

    records = []
    sid = 1
    for alpha in grid:
        for beta in grid:
            mdl = gen_model(y_true, alpha, seed=1000 + sid)
            obs = gen_obs(y_true, beta, seed=2000 + sid)

            _, mu_tl = tl_ll_mu(mdl)
            _, mu_el = el_ll_mu(obs)

            mse_tl = mse(y_true, mu_tl)
            mse_el = mse(y_true, mu_el)

            best_k1, best_mse = 0.0, 1e18
            for k1 in np.linspace(0, 1, 101):
                cur = mse(y_true, k1 * mu_tl + (1 - k1) * mu_el)
                if cur < best_mse:
                    best_k1, best_mse = k1, cur

            records.append(
                dict(Dist=name, alpha=alpha, beta=beta,
                     mu_tl=mu_tl, mu_el=mu_el,
                     MSE_TL=mse_tl, MSE_EL=mse_el,
                     k1_opt=best_k1, MSE_Fused=best_mse)
            )
            sid += 1
    return pd.DataFrame(records)


def main() -> None:
    """Run all four scenarios with fixed α, β grid and save CSV."""
    df = pd.concat([run_scenario(i) for i in range(1, 5)], ignore_index=True)
    outfile = Path("sim_fixed_alpha_beta.csv")
    df.to_csv(outfile, index=False)
    print(f"[Simulation] finished → {outfile} ({len(df)} rows)")


if __name__ == "__main__":
    main()
