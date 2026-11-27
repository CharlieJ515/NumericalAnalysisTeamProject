"""범위 안(0~10) 적합 방법 비교 + 외삽 모습 + 미분 추정 (분석용. scipy/sklearn 사용, 제출 불가)."""
import time, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

tr = np.genfromtxt("data/train.csv", delimiter=",", names=True)
x, y = tr["x"], tr["y"]
rng = np.random.default_rng(42)
NODES = np.array([0.0, 0.7781, 1.5720, 3.2125, 4.5703, 5.6815, 7.0363, 8.0495, 9.2934, 10.0])

def fit_leg20(xt, yt):
    c = L.legfit((xt - 5) / 5, yt, 20); return lambda q: L.legval((q - 5) / 5, c)

def fit_pw3(xt, yt):
    cs = []
    for a, b in zip(NODES[:-1], NODES[1:]):
        m = (xt >= a) & (xt <= b); B = np.vander(xt[m] - (a + b) / 2, 4)
        cs.append(np.linalg.solve(B.T @ B, B.T @ yt[m]))   # 구간별 정규방정식
    def f(q):
        k = np.clip(np.searchsorted(NODES, q) - 1, 0, len(cs) - 1)
        mid = (NODES[k] + NODES[k + 1]) / 2
        return np.array([np.polyval(cs[i], qi - mi) for i, qi, mi in zip(k, q, mid)])
    return f

def fit_spline(xt, yt):
    o = np.argsort(xt); xs, ys = xt[o], yt[o]
    keep = np.r_[True, np.diff(xs) > 0]
    return UnivariateSpline(xs[keep], ys[keep], k=3, s=keep.sum() * 0.2**2)

def fit_knn(xt, yt):
    m = KNeighborsRegressor(200).fit(xt[:, None], yt); return lambda q: m.predict(q[:, None])
def fit_gbm(xt, yt):
    m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(xt[:, None], yt); return lambda q: m.predict(q[:, None])
def fit_mlp(xt, yt):
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=300, random_state=0).fit(xt[:, None] / 5 - 1, yt)
    return lambda q: m.predict(q[:, None] / 5 - 1)

methods = {"20th Legendre LS": fit_leg20, "piecewise cubic (9 seg, LS)": fit_pw3,
           "smoothing spline": fit_spline, "kNN (k=200)": fit_knn,
           "gradient boosting": fit_gbm, "MLP 64x64 tanh": fit_mlp}

folds = np.array_split(rng.permutation(len(x)), 5)
full = {}
for name, fit in methods.items():
    t = time.time(); r = []
    for i in range(5):
        va = folds[i]; trn = np.concatenate([folds[j] for j in range(5) if j != i])
        f = fit(x[trn], y[trn]); r.append(np.sqrt(np.mean((f(x[va]) - y[va]) ** 2)))
    full[name] = fit(x, y)
    print(f"{name:28s} CV RMSE {np.mean(r):.4f} ± {np.std(r):.4f}   ({time.time()-t:.1f}s)  "
          f"f(10,11,12) = {np.round(full[name](np.array([10., 11., 12.])), 2)}")

# 500개 샘플 정규방정식 초기값의 질
for name in ["20th Legendre LS", "piecewise cubic (9 seg, LS)"]:
    rs = []
    for s in range(20):
        idx = np.random.default_rng(s).choice(len(x), 500, replace=False)
        f = methods[name](x[idx], y[idx]); rs.append(np.sqrt(np.mean((f(x) - y) ** 2)))
    print(f"500-sample init  {name:28s} full-data RMSE median {np.median(rs):.4f}, worst {np.max(rs):.4f}")

# 그림
q = np.linspace(0, 12, 1200)
fig, (a, b) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
a.scatter(x[::5], y[::5], s=1, alpha=0.1, color="gray")
for name, f in full.items():
    a.plot(q, f(q), lw=1.4, label=name)
a.axvspan(10, 12, color="orange", alpha=0.15); a.set_ylim(-8, 6); a.legend(fontsize=8, loc="upper left")
a.set_title("in-range fits (all ≈ noise floor) vs. what they do beyond x = 10"); a.grid(alpha=.3)

# 미분: 원자료 차분(노이즈 폭증) vs 구간 평균 차분 vs 매끄러운 곡선의 미분
o = np.argsort(x); xs, ys = x[o], y[o]
raw = np.diff(ys) / np.diff(xs); keep = np.isfinite(raw) & (np.abs(raw) < 1e5)
b.scatter(xs[1:][keep][::50], raw[keep][::50], s=1, alpha=0.2, color="gray", label="raw neighbor difference (h≈0.0002)")
for h in [0.25]:
    e = np.arange(0, 10 + h, h); cm = (e[:-1] + e[1:]) / 2
    mb = np.array([y[(x >= lo) & (x < hi)].mean() for lo, hi in zip(e[:-1], e[1:])])
    b.plot((cm[1:] + cm[:-1]) / 2, np.diff(mb) / h, "o-", ms=3, label=f"bin-mean difference (h={h})")
sp = full["smoothing spline"]
b.plot(q[q <= 10], sp.derivative()(q[q <= 10]), lw=2, color="black", label="smoothing spline f'(x)")
b.axhline(0, color="k", lw=0.5); b.set_ylim(-6, 6); b.set_xlabel("x"); b.set_ylabel("f'(x)")
b.legend(fontsize=8, loc="lower left"); b.grid(alpha=.3); b.axvspan(10, 12, color="orange", alpha=0.15)
fig.tight_layout(); fig.savefig("analysis/jsm/figures/inrange_methods.png", dpi=120)
print("raw diff std:", np.std(raw[keep]).round(1), " spline f'(10):", float(sp.derivative()(10.0)).__round__(3),
      " f''(10):", float(sp.derivative(2)(10.0)).__round__(3))
