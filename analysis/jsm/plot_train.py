"""train.csv 산점도 + 20차 Legendre 최소제곱 곡선의 극점·변곡점 표시 (분석용, 최종 모델 아님)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L

tr = np.genfromtxt("data/train.csv", delimiter=",", names=True)
te = np.genfromtxt("data/test.csv", delimiter=",", names=True)
x, y = tr["x"], tr["y"]

# 노이즈 추정 1: 모델 없이 x로 정렬한 이웃 차분 (f가 매끄러우면 Var(y_{i+1}-y_i) ≈ 2σ²)
o = np.argsort(x); d = np.diff(y[o])
sigma_diff = np.sqrt(np.mean(d**2) / 2)

# 20차 Legendre 최소제곱 (z = (x-5)/5 ∈ [-1,1])
z = (x - 5) / 5
c = L.legfit(z, y, 20)
res = y - L.legval(z, c)
sigma_fit = res.std()

def real_roots(cc):
    r = L.legroots(cc)
    r = r[np.isreal(r)].real
    return np.sort(r[(r >= z.min()) & (r <= z.max())])

c1, c2, c3 = L.legder(c, 1), L.legder(c, 2), L.legder(c, 3)
ext = real_roots(c1)
inf = real_roots(c2)
inf = inf[np.abs(L.legval(inf, c3)) > 1e-8]  # 부호가 실제로 바뀌는 점만
ext_x, inf_x = ext * 5 + 5, inf * 5 + 5
print(f"sigma (neighbor diff) = {sigma_diff:.4f}, sigma (20th-order residual) = {sigma_fit:.4f}")
print("extrema   :", np.round(ext_x, 4))
print("inflection:", np.round(inf_x, 4))

xs = np.linspace(0, 10, 2000); fs = L.legval((xs - 5) / 5, c)
fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
a.scatter(x, y, s=1, alpha=0.12, color="tab:blue", label="train (50,000)")
a.plot(xs, fs, color="black", lw=1.5, label="20th-order least squares (analysis only)")
a.scatter(ext_x, L.legval(ext, c), marker="^", s=70, color="tab:green", zorder=5, label="extrema f'=0")
a.scatter(inf_x, L.legval(inf, c), marker="o", s=70, color="tab:red", zorder=5, label="inflection f''=0")
for xi in inf_x:
    a.axvline(xi, color="tab:red", ls=":", lw=0.8)
    a.text(xi, a.get_ylim()[0] if False else -4.6, f"{xi:.2f}", color="tab:red", ha="center", fontsize=8, rotation=90, va="bottom")
for ax in (a, b):
    ax.axvspan(10, 12, color="orange", alpha=0.15)
a.text(11, 4.6, "extrapolation\n(test only, y unknown)", ha="center", fontsize=9)
a.set_ylim(-4.8, 5.4)
a.set_ylabel("y"); a.set_title(f"train.csv  |  noise σ ≈ {sigma_diff:.3f} (neighbor diff), {sigma_fit:.3f} (residual)")
a.legend(loc="upper left", fontsize=8); a.grid(alpha=.3)
bins = np.arange(0, 12.25, 0.25)
b.hist(x, bins=bins, alpha=.6, label="train x"); b.hist(te["x"], bins=bins, alpha=.6, label="test x")
b.set_xlabel("x"); b.set_ylabel("count"); b.legend(); b.grid(alpha=.3); b.set_xlim(0, 12)
fig.tight_layout(); fig.savefig("analysis/jsm/figures/train_scatter.png", dpi=130)
print("saved analysis/jsm/figures/train_scatter.png")
