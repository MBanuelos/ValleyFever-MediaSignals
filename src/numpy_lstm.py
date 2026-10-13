"""
Minimal single-layer LSTM regressor in pure NumPy (manual back-propagation through time).

Why NumPy: it needs no deep-learning framework, so the whole repo runs with
numpy/pandas/scikit-learn/statsmodels/xgboost. The architecture matches the repo's
PyTorch notebook (LSTM -> last hidden state -> linear), except that it predicts all
forecast steps at once (multi-output) instead of one step.

Input  X : (N, T, D)  sequences of standardized features
Target Y : (N, K)     K future values (here the next 6 monthly log-changes)
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyLSTM:
    def __init__(self, n_features, n_outputs, hidden=24, seed=5):
        rng = np.random.default_rng(seed)
        H, D = hidden, n_features
        k = 1.0 / np.sqrt(H)
        self.H, self.D, self.K = H, D, n_outputs
        self.p = {
            "Wx": rng.uniform(-k, k, (D, 4 * H)),
            "Wh": rng.uniform(-k, k, (H, 4 * H)),
            "b": np.zeros(4 * H),
            "Wy": rng.uniform(-k, k, (H, n_outputs)),
            "by": np.zeros(n_outputs),
        }
        self.p["b"][H:2 * H] = 1.0            # forget-gate bias
        self.rng = rng

    # ---- forward -----------------------------------------------------------
    def _forward(self, X):
        N, T, _ = X.shape
        H = self.H
        h = np.zeros((N, H)); c = np.zeros((N, H))
        cache = []
        for t in range(T):
            z = X[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
            i = _sigmoid(z[:, :H]); f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = _sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((X[:, t], h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        out = h @ self.p["Wy"] + self.p["by"]
        return out, h, cache

    def predict(self, X):
        return self._forward(np.asarray(X, dtype=float))[0]

    # ---- loss + gradients --------------------------------------------------
    def loss_and_grads(self, X, Y, l2=0.0):
        N = X.shape[0]
        out, hT, cache = self._forward(X)
        diff = out - Y
        loss = np.mean(diff ** 2) + 0.5 * l2 * sum(np.sum(self.p[k] ** 2) for k in ("Wx", "Wh", "Wy"))
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dout = 2.0 * diff / diff.size
        g["Wy"] = hT.T @ dout
        g["by"] = dout.sum(0)
        dh = dout @ self.p["Wy"].T
        dc = np.zeros_like(dh)
        H = self.H
        for (x, h_prev, c_prev, i, f, gg, o, tc) in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc ** 2)
            di = dc * gg
            dg = dc * i
            df = dc * c_prev
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - gg ** 2), do * o * (1 - o)], axis=1)
            g["Wx"] += x.T @ dz
            g["Wh"] += h_prev.T @ dz
            g["b"] += dz.sum(0)
            dh = dz @ self.p["Wh"].T
            dc = dc * f
        for k in ("Wx", "Wh", "Wy"):
            g[k] += l2 * self.p[k]
        return loss, g

    # ---- training ----------------------------------------------------------
    def fit(self, X, Y, epochs=60, lr=0.01, batch=16, l2=1e-4, clip=1.0):
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        N = len(X)
        for _ in range(epochs):
            order = self.rng.permutation(N)
            for s in range(0, N, batch):
                idx = order[s:s + batch]
                _, grads = self.loss_and_grads(X[idx], Y[idx], l2)
                norm = np.sqrt(sum(np.sum(gr ** 2) for gr in grads.values()))
                scale = min(1.0, clip / (norm + 1e-12))
                step += 1
                for k in self.p:
                    gk = grads[k] * scale
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    v[k] = b2 * v[k] + (1 - b2) * gk ** 2
                    mh = m[k] / (1 - b1 ** step); vh = v[k] / (1 - b2 ** step)
                    self.p[k] -= lr * mh / (np.sqrt(vh) + eps)
        return self


if __name__ == "__main__":
    # numerical gradient check
    rng = np.random.default_rng(0)
    net = NumpyLSTM(4, 3, hidden=5, seed=1)
    X = rng.normal(size=(7, 6, 4)); Y = rng.normal(size=(7, 3))
    _, grads = net.loss_and_grads(X, Y, l2=1e-3)
    worst = 0.0
    for name, arr in net.p.items():
        flat = arr.reshape(-1)
        for j in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + 1e-6; lp, _ = net.loss_and_grads(X, Y, l2=1e-3)
            flat[j] = old - 1e-6; lm, _ = net.loss_and_grads(X, Y, l2=1e-3)
            flat[j] = old
            num = (lp - lm) / 2e-6
            ana = grads[name].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    print("max relative gradient error:", worst)
    # learnability check
    net = NumpyLSTM(2, 1, hidden=12, seed=2)
    Xs = rng.normal(size=(200, 6, 2)); Ys = Xs[:, -1, :1] * 0.8 - Xs[:, -3, 1:2] * 0.5
    net.fit(Xs, Ys, epochs=80)
    print("train MSE after fit:", float(np.mean((net.predict(Xs) - Ys) ** 2)), "(target variance", float(Ys.var()), ")")
