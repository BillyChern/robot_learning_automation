"""Reset-to-states curriculum in the style of SGS-RL (success-guided sampling): a bank of start configurations
(domain parameters + scripted pre-roll to a partly inserted / misaligned / jammed state), sampled with a Beta-shaped
weight on each configuration's recent success rate (peak at 0.5), mixed with fresh episodes from the eval distribution."""
import numpy as np

from . import params as P
from .policy import H_STEPS


class Curriculum:
    def __init__(self, size=4096, fresh=0.4, kappa=4.0, eps=0.05, seed=0, variants=None, v2=None):
        self.rng, self.v2 = np.random.default_rng(seed), v2
        self.variants = variants or dict(carry=0.8, aligned=0.2)   # start variants (params.VARIANTS) and shares
        self.size, self.fresh, self.kappa, self.eps = size, fresh, kappa, eps
        self.next_seed = 10**6 * (seed + 1)
        self.cfg = [self.new() for _ in range(size)]
        self.p = np.full(size, 0.5)
        self.n = np.zeros(size, int)

    def new(self):
        """One bank entry: params seed, start variant, pre-roll kind and length."""
        self.next_seed += 1
        kind = self.rng.choice([None, "push", "drift"], p=[0.4, 0.3, 0.3])
        return dict(seed=self.next_seed, variant=self.variant(), kind=kind,
                    k_pre=0 if kind is None else int(self.rng.integers(2, H_STEPS // 2)))

    def variant(self):
        names = list(self.variants)
        w = np.array([self.variants[n] for n in names], float)
        return names[self.rng.choice(len(names), p=w / w.sum())]

    def weights(self):
        w = (self.p + self.eps) ** (0.5 * self.kappa) * (1 - self.p + self.eps) ** (0.5 * self.kappa)
        w[self.n == 0] = w.max()
        return w / w.sum()

    def sample(self, count):
        """count starts: dict(p, kind, k_pre, idx) with idx -1 for fresh episodes."""
        n_fresh = int(round(self.fresh * count))
        idx = np.r_[np.full(n_fresh, -1), self.rng.choice(self.size, count - n_fresh, p=self.weights())]
        out = []
        for i in idx:
            c = dict(seed=int(self.rng.integers(2**31)), variant=self.variant(), kind=None, k_pre=0) if i < 0 else self.cfg[i]
            p = P.sample(np.random.default_rng(c["seed"]), c["variant"], self.v2)
            out.append(dict(p=p, kind=c["kind"], k_pre=c["k_pre"], idx=int(i)))
        return out

    def update(self, idx, success):
        """Record outcomes; replace mastered (p > 0.95) and hopeless (p < 0.02 after 12 visits) entries."""
        for i, s in zip(idx, success):
            if i < 0:
                continue
            self.n[i] += 1
            self.p[i] += (0.5 if self.n[i] == 1 else 0.25) * (float(s) - self.p[i])
            if (self.n[i] >= 4 and self.p[i] > 0.95) or (self.n[i] >= 12 and self.p[i] < 0.02):
                self.cfg[i], self.p[i], self.n[i] = self.new(), 0.5, 0

    def stats(self):
        seen = self.n > 0
        return dict(bank_p=float(self.p[seen].mean()) if seen.any() else 0.5, bank_seen=int(seen.sum()))
