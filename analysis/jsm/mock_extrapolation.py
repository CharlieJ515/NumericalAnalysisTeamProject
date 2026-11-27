"""모의 외삽(rolling origin) 검증: x < c 로만 학습 → [c, c+h] 예측. 분석용(lstsq), 최종 모델 아님.
점수는 원자료 y 기준 RMSE이므로 노이즈 하한 ≈ 0.20 이 포함된다."""
import numpy as np
from numpy.polynomial import legendre as L
t = np.genfromtxt("data/train.csv", delimiter=",", names=True); x, y = t["x"], t["y"]

def win_poly(c, w, d):
    m = (x >= c - w) & (x < c); cf = np.polyfit(x[m] - c, y[m], d)  # 분석용
    return lambda q: np.polyval(cf, q - c)
def damped(c, w, k):
    cf = np.polyfit(x[(x >= c - w) & (x < c)] - c, y[(x >= c - w) & (x < c)], 1); b, a = cf
    return lambda q: a + b * (1 - np.exp(-k * (q - c))) / k
def glob_leg(c, M):
    m = x < c; z = lambda v: 2 * v / c - 1; cf = L.legfit(z(x[m]), y[m], M)
    return lambda q: L.legval(z(q), cf)

methods = {"flat (last 0.3 mean)": lambda c: win_poly(c, 0.3, 0)}
for w in (0.3, 0.5, 1.0): methods[f"line w={w}"] = (lambda w: lambda c: win_poly(c, w, 1))(w)
for w in (1.0, 2.0): methods[f"quad w={w}"] = (lambda w: lambda c: win_poly(c, w, 2))(w)
for k in (0.5, 1.0): methods[f"damped line w=0.3 k={k}"] = (lambda k: lambda c: damped(c, 0.3, k))(k)
for M in (5, 8, 12, 20): methods[f"global Legendre M={M}"] = (lambda M: lambda c: glob_leg(c, M))(M)

setups = [("h=2", [6.0, 6.5, 7.0, 7.5, 8.0], 2.0), ("h=1", [7.0, 7.5, 8.0, 8.5, 9.0], 1.0)]
for tag, origins, h in setups:
    print(f"\n== horizon {tag}, origins {origins} (RMSE incl. noise 0.20) ==")
    rows = []
    for name, mk in methods.items():
        r = []
        for c in origins:
            f = mk(c); m = (x >= c) & (x < c + h); r.append(np.sqrt(np.mean((f(x[m]) - y[m]) ** 2)))
        rows.append((np.mean(r), name, r))
    for mean, name, r in sorted(rows):
        print(f"{name:26s} mean {mean:7.3f} | " + " ".join(f"{v:7.2f}" for v in r))
