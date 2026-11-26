# 제출 기록

- 제출 파일: `submissions/YYYYMMDD_번호_이름.csv`
- 그림: `uv run python plot_submission.py <csv...>` → `submissions/figures/<파일이름>.png`, 여러 개를 겹칠 때는 `--overlay <이름>`

## 제출 목록

| # | 날짜 | 파일 | 생성 브랜치 (커밋) | x ≤ 10 | x > 10 | public RMSE |
|---|---|---|---|---|---|---|
| 1 | 2026-11-26 | `20261126_1_phase1_flat.csv` | `exp/model-selection` (`5934084`) | 구간별 3차식 | −2.1440 | 0.88862 |
| 2 | 2026-11-26 | `20261126_2_phase1_up.csv` | `exp/model-selection` (`5934084`) | 구간별 3차식 | −2.1440 + 1·(x − 10) | 1.40438 |
| 3 | 2026-11-26 | `20261126_3_phase1_down.csv` | `exp/model-selection` (`5934084`) | 구간별 3차식 | −2.1440 − 1·(x − 10) | |

## 2026-11-26 phase 1

### 공통: x ≤ 10 예측
- 모델: 구간별 3차 다항식, 변곡점 기준 분할, 폭 0.2 미만 구간 병합, 경계 겹침 δ = 0.1
  - `uv run python -m analysis.piecewise_cubic --nodes inflection --min-width 0.2 --overlap 0.1 --save-predictions`
  - 노드: 0.0003, 0.7781, 1.5720, 3.2125, 4.5703, 5.6815, 7.0363, 8.0495, 9.2934, 10.0000
  - 구간마다 `PolynomialModel(degree=3)`, MSE, Adam lr 0.05, full-batch 1000 step, 초기값 0
  - train RMSE 0.2011, 노드 점프 평균 0.0070 / 최대 0.0217
- 세 제출 파일의 x ≤ 10 예측은 비트 단위로 같음

### x > 10 예측
- `uv run python -m analysis.probe_phase1`
- 시작값 f(10) = −2.1440: train x ≥ 9.7 구간 직선 회귀의 x = 10 값 (같은 회귀의 기울기 −2.1749)
- 기울기 0, +1, −1 직선 (제출 순서: flat, up, down)

![phase 1](figures/20261126_phase1.png)

### 결과

| 파일 | public RMSE |
|---|---|
| `20261126_1_phase1_flat.csv` | 0.88862 |
| `20261126_2_phase1_up.csv` | 1.40438 |
| `20261126_3_phase1_down.csv` | |
