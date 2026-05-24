from __future__ import annotations



"""

KlasicniRegresori_v2.py — Loto 7/39 predikcija (potpuno determinističko)

  • Vremenski tačan split (bez shuffle), TimeSeriesSplit za HP tjuning.
  • Cilj nije „broj po poziciji" (to konvergira ka prosecima kolona),
    nego MULTI-LABEL skor za svaki broj 1..39 → uzimamo top-7.
  • Feature engineering:
      - lag prozor poslednjih L izvlačenja (flatten),
      - rolling frekvencije po brojevima (W = 20/50/100),
      - gap (koliko kola od poslednjeg pojavljivanja za svaki broj),
      - statistike prošlog kola (suma, parnost, raspon, low/high).
  • HP tjuning sa TimeSeriesSplit i custom scorer-om
    (label_ranking_average_precision) za RFR i XGB.
  • Ansambl: DT + KNN + RF + Ridge + GBR + XGBoost (svi multi-output).
  • Stacking nadgradnja: meta-Ridge nad skorovima baznih modela.
  • Post-processing: 7 jedinstvenih, sortirano, u opsegu 1..39.
  • Back-test: prosečni pogoci po modelu, hit-rate %, ROC AUC, MAP.
  • Snimanje rezultata u TXT sa timestamp-om.
  • Determinizam: PYTHONHASHSEED, np/random/algorithm_globals = SEED,
    svi modeli random_state=SEED, n_jobs=1, jedna BLAS nit.

Pokretanje:
    python KlasicniRegresori_v2.py






HP tjuning sa TimeSeriesSplit(n_splits=3) i custom scorer-om label_ranking_average_precision (LRAP) — pravo merilo za multi-label ranking — za RFR (n_estimators, max_depth, min_samples_leaf) i XGB (n_estimators, max_depth, learning_rate). Prekidač DO_TUNE = True/False ako želiš brži run.
Stacking: meta-Ridge nad skorovima svih baznih modela (treniran na poslednjih 20% trening seta) — pored proseka ansambla dobijaš i stacking predikciju.
Bogatije metrike u back-testu: hits/7, hit%, ROC AUC (macro), LRAP — po modelu, ansamblu i stackingu. Plus slučajan baseline ≈ 1.256 za poređenje.
Opis kombinacije: suma, broj neparnih, broj niskih (≤19), raspon.
Snimanje u TXT: KlasicniRegresori_v2_predikcija.txt — append sa timestamp-om, seed-om i svim modelima.
Determinizam i dalje hard: PYTHONHASHSEED, BLAS 1 nit, n_jobs=1 svuda, random_state=SEED svuda → isto pokretanje → ista kombinacija.





Napomena o vremenu: 
GridSearchCV sa RFR i XGB nad ~4400 redova x 3 fold-a će trajati par minuta (možda 5-15 zavisno od mašine). 
Ako želiš odmah brzo bez tjuninga, promeni DO_TUNE = False na vrhu fajla.





Raspored:
prvih 100 redova → samo "zagrevanje" rolling prozora (W=20/50/100) i gap-a (features iz prošlosti)
4420 parova (X, y) → trening modela + HP tjuning
poslednjih 100 kola → back-test (ROC AUC / LRAP / hits/7)
features iz poslednjeg (4619.) reda CSV-a → predikcija sledećeg, još neodigranog kola
Nijedno od 4620 izvlačenje nije izbačeno — sve ulazi ili kao input-feature ili kao target ili kao test.





GBR sa MultiOutputRegressor fituje 39 nezavisnih GBR modela (po jedan za svaki broj 1-39), svaki sa 200 stabala. Realno na Mac-u to traje 1-3 min. 
Na MacBook Pro, M1, 16GB RAM, sve zavrseno za 8 minuta.

Posle GBR ide samo XGB, koji je 5-10x brži od GBR jer ima tree_method="hist".

Ako bude previse sporo moze da se smanji n_estimators na 100 za GBR ili da se potpuno izbaci (RFR + XGB ionako pokrivaju istu rolu).

"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

from datetime import datetime
import pytz

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    label_ranking_average_precision_score,
    roc_auc_score,
    make_scorer,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from qiskit_machine_learning.utils import algorithm_globals
    algorithm_globals.random_seed = SEED
except Exception:
    pass

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# Konfiguracija
# ============================================================
CSV_PATH        = "/KvantniRegresor/loto7_4620_k41.csv"
OUT_TXT         = "/KvantniRegresor/KlasicniRegresori_v2_predikcija.txt"
N_MIN, N_MAX    = 1, 39
K               = 7
LAG             = 5
WINDOWS         = (20, 50, 100)
BACKTEST_N      = 100
TUNE_SPLITS     = 3         # TimeSeriesSplit n_splits za GridSearchCV
DO_TUNE         = False     # postavi na False za bržu (default) konfiguraciju


def stamp() -> str:
    return datetime.now(pytz.timezone("Europe/Belgrade")).strftime("%d.%m.%Y_%H.%M.%S")


print()
print("🔁 KlasicniRegresori_v2 — start ", stamp())
print()


# ============================================================
# 1) Učitavanje CSV-a (bez headera, 7 kolona)
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
df = df.iloc[:, :K].astype(int)
draws = df.values  # shape (N, 7)
N = draws.shape[0]
print(f"✅ CSV učitan: {CSV_PATH}")
print(f"   broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


# ============================================================
# 2) Multi-hot reprezentacija svakog izvlačenja (N, 39)
# ============================================================
def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    M = rows.shape[0]
    out = np.zeros((M, N_MAX), dtype=np.int8)
    for i in range(M):
        for v in rows[i]:
            if N_MIN <= v <= N_MAX:
                out[i, v - 1] = 1
    return out


Y_full = draws_to_multihot(draws)  # (N, 39)


# ============================================================
# 3) Feature engineering — sve features se računaju samo iz prošlosti
# ============================================================
def build_features(draws_arr: np.ndarray,
                   y_multi: np.ndarray,
                   lag: int = LAG,
                   windows=WINDOWS) -> np.ndarray:
    n, _ = draws_arr.shape
    feats = []

    # 3.1 Lag prozor: poslednjih `lag` kola, flatten (lag*7)
    for L in range(1, lag + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[L:] = draws_arr[:-L]
        feats.append(shifted)
    lag_block = np.concatenate(feats, axis=1)

    # 3.2 Rolling frekvencije po broju (cumulative trick)
    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for W in windows:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:W + 1] = cum[:W]
        rolled[W + 1:] = cum[W:-1] - cum[:-W - 1]
        rolling_blocks.append(rolled / float(W))
    roll_block = np.concatenate(rolling_blocks, axis=1)

    # 3.3 Gap: za svaki broj, koliko kola od zadnjeg pojavljivanja
    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    # 3.4 Statistike prošlog kola
    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum  = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd  = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low  = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng  = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stat_block = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    X = np.concatenate([lag_block, roll_block, gap, stat_block], axis=1)
    return X


X_full = build_features(draws, Y_full)
print(f"✅ Features: X_full.shape = {X_full.shape}, Y_full.shape = {Y_full.shape}")
print()

START = max(LAG, max(WINDOWS))


# ============================================================
# 4) Konstrukcija (X, y) parova
# ============================================================
X_all = X_full[START:N].astype(float)
Y_all = Y_full[START:N].astype(float)
print(f"   trening domen: {X_all.shape[0]} parova")
print()


# ============================================================
# 5) Vremenski split + skaliranje
# ============================================================
n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_back,  Y_back  = X_all[n_train:], Y_all[n_train:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_back_s  = scaler.transform(X_back)

X_next_raw = X_full[N - 1:N].astype(float)
X_next_s   = scaler.transform(X_next_raw)


# ============================================================
# 6) Custom scorer za HP tjuning (multi-label ranking)
# ============================================================
def lrap_scorer_fn(y_true, y_score):
    # GridSearchCV očekuje viši = bolji
    y_true_i = (np.asarray(y_true) > 0.5).astype(int)
    return label_ranking_average_precision_score(y_true_i, np.asarray(y_score))

LRAP_SCORER = make_scorer(lrap_scorer_fn, greater_is_better=True)


# ============================================================
# 7) HP tjuning sa TimeSeriesSplit (samo za RFR i XGB)
# ============================================================
tscv = TimeSeriesSplit(n_splits=TUNE_SPLITS)
tuned = {}

if DO_TUNE:
    print("🔧 HP tjuning (TimeSeriesSplit, scorer=LRAP) ...")

    # RFR
    rfr_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=SEED, n_jobs=1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth":    [8, 12, None],
            "min_samples_leaf": [1, 3],
        },
        scoring=LRAP_SCORER, cv=tscv, n_jobs=1, refit=True,
    )
    rfr_grid.fit(X_train_s, Y_train)
    tuned["RFR"] = rfr_grid.best_estimator_
    print(f"   RFR best: {rfr_grid.best_params_}  (LRAP={rfr_grid.best_score_:.4f})")

    # XGB
    if HAS_XGB:
        base_xgb = xgb.XGBRegressor(
            random_state=SEED, n_jobs=1, verbosity=0,
            tree_method="hist", subsample=0.9, colsample_bytree=0.9,
        )
        xgb_grid = GridSearchCV(
            estimator=MultiOutputRegressor(base_xgb),
            param_grid={
                "estimator__n_estimators":   [200, 400],
                "estimator__max_depth":      [3, 5],
                "estimator__learning_rate":  [0.05, 0.1],
            },
            scoring=LRAP_SCORER, cv=tscv, n_jobs=1, refit=True,
        )
        xgb_grid.fit(X_train_s, Y_train)
        tuned["XGB"] = xgb_grid.best_estimator_
        print(f"   XGB best: {xgb_grid.best_params_}  (LRAP={xgb_grid.best_score_:.4f})")
    print()


# ============================================================
# 8) Definicija svih modela
# ============================================================
def make_models():
    models = {
        "DTR": DecisionTreeRegressor(random_state=SEED, max_depth=10, min_samples_leaf=4),
        "KNN": KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=1),
        "RFR": tuned.get("RFR", RandomForestRegressor(
                  n_estimators=400, max_depth=10, random_state=SEED, n_jobs=1)),
        "LR":  Ridge(alpha=1.0, random_state=SEED),
        "GBR": MultiOutputRegressor(
                  GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED)),
    }
    if HAS_XGB:
        models["XGB"] = tuned.get("XGB", MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=SEED, n_jobs=1, verbosity=0, tree_method="hist",
            )
        ))
    return models


print("⚛️ Treniranje modela ...")
models = make_models()
for name, m in models.items():
    if name in tuned:
        print(f"   ✅ {name} (već fit-ovan kroz GridSearchCV).")
        continue
    m.fit(X_train_s, Y_train)
    print(f"   ✅ {name} treniran.")
print()


# ============================================================
# 9) Top-K iz skorova: 7 jedinstvenih, sortirano, 1..39
# ============================================================
def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    s = np.asarray(scores_1d, dtype=float).copy()
    order = np.lexsort((np.arange(N_MAX), -s))
    chosen = order[:k] + 1
    return np.sort(chosen)


# ============================================================
# 10) Stacking: meta-Ridge nad skorovima baznih modela
#     - X_meta_train: skorovi baznih modela na (rolling) trening delu
#       — koristimo poslednje 20% trening seta kao "in-sample" za meta
#     - alternativno: nested CV; ovde radimo jednostavniji vremenski split
# ============================================================
print("🪡 Stacking (meta-Ridge) ...")
n_meta = max(200, n_train // 5)
X_meta_in,  Y_meta_in  = X_train_s[-n_meta:], Y_train[-n_meta:]
base_scores_meta = np.concatenate(
    [m.predict(X_meta_in) for m in models.values()], axis=1
)  # (n_meta, 39 * n_models)
meta = Ridge(alpha=1.0, random_state=SEED)
meta.fit(base_scores_meta, Y_meta_in)
print(f"   meta-Ridge fitovan na poslednjih {n_meta} trening uzoraka.")
print()


# ============================================================
# 11) Back-test: pogoci po modelu + ensemble + stacking + AUC + LRAP
# ============================================================
print(f"🧪 Back-test (poslednjih {BACKTEST_N} izvlačenja):")
all_scores = {name: m.predict(X_back_s) for name, m in models.items()}
ensemble_scores = np.mean(np.stack(list(all_scores.values()), axis=0), axis=0)
stack_input_back = np.concatenate(list(all_scores.values()), axis=1)
stack_scores = meta.predict(stack_input_back)

def avg_hits(scores_2d, Y):
    h = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(Y[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        h += len(true_set & pred_set)
    return h / scores_2d.shape[0]

def safe_auc(Y, scores):
    try:
        return roc_auc_score(Y, scores, average="macro")
    except Exception:
        return float("nan")

def safe_lrap(Y, scores):
    try:
        return label_ranking_average_precision_score(Y.astype(int), scores)
    except Exception:
        return float("nan")

rows = []
for name, s in all_scores.items():
    rows.append((name, avg_hits(s, Y_back), safe_auc(Y_back, s), safe_lrap(Y_back, s)))
rows.append(("ENSEMBLE", avg_hits(ensemble_scores, Y_back),
             safe_auc(Y_back, ensemble_scores), safe_lrap(Y_back, ensemble_scores)))
rows.append(("STACK",    avg_hits(stack_scores,    Y_back),
             safe_auc(Y_back, stack_scores),    safe_lrap(Y_back, stack_scores)))

print(f"   {'model':<9} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
for name, hits, auc, lrap in rows:
    print(f"   {name:<9} {hits:>8.3f} {100*hits/K:>6.1f}% {auc:>7.3f} {lrap:>7.3f}")
print()
# Slučajan baseline za orijentaciju: 7 * 7/39 = 1.256
print(f"   (slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


# ============================================================
# 12) Prava predikcija SLEDEĆEG (još neodigranog) kola
# ============================================================
print("🎯 Predikcija SLEDEĆEG kola (po modelu):")
next_scores = {name: m.predict(X_next_s)[0] for name, m in models.items()}
preds = {name: topk_from_scores(s) for name, s in next_scores.items()}
for name, p in preds.items():
    print(f"   {name:<5} -> {p.tolist()}")

ensemble_next = np.mean(np.stack(list(next_scores.values()), axis=0), axis=0)
ensemble_pick = topk_from_scores(ensemble_next)

stack_in_next = np.concatenate(
    [s.reshape(1, -1) for s in next_scores.values()], axis=1
)
stack_next   = meta.predict(stack_in_next)[0]
stack_pick   = topk_from_scores(stack_next)

print()
print(f"🏁 ANSAMBL (prosek skorova): {ensemble_pick.tolist()}")
print(f"🏁 STACKING (meta-Ridge)  : {stack_pick.tolist()}")
print()


# ============================================================
# 13) Validacija + opis kombinacije + snimanje u TXT
# ============================================================
def describe(pick: np.ndarray) -> str:
    s = pick.sum()
    odd = int((pick % 2 == 1).sum())
    low = int((pick <= 19).sum())
    rng = int(pick.max() - pick.min())
    return f"suma={s}, neparnih={odd}/{K}, niskih(≤19)={low}/{K}, raspon={rng}"

for name, p in [("ANSAMBL", ensemble_pick), ("STACKING", stack_pick)]:
    assert len(set(p.tolist())) == K
    assert p.min() >= N_MIN and p.max() <= N_MAX
    assert list(p) == sorted(p.tolist())
    print(f"✅ {name} validan ({describe(p)}).")

with open(OUT_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n--- {stamp()} (seed={SEED}, N={N}) ---\n")
    for name, p in preds.items():
        f.write(f"{name:<5} -> {p.tolist()}\n")
    f.write(f"ENSEMBLE -> {ensemble_pick.tolist()}  ({describe(ensemble_pick)})\n")
    f.write(f"STACK    -> {stack_pick.tolist()}  ({describe(stack_pick)})\n")
print(f"📝 Snimljeno u: {OUT_TXT}")

print()
print("🔁 KlasicniRegresori_v2 — stop ", stamp())
print()



"""

🔁 KlasicniRegresori_v2 — start  24.05.2026_09.54.24

✅ CSV učitan: /KvantniRegresor/loto7_4620_k41.csv
   broj izvlačenja: 4620, brojeva po kolu: 7

✅ Features: X_full.shape = (4620, 195), Y_full.shape = (4620, 39)

   trening domen: 4520 parova

⚛️ Treniranje modela ...
   ✅ DTR treniran.
   ✅ KNN treniran.
   ✅ RFR treniran.
   ✅ LR treniran.
   ✅ GBR treniran.
   ✅ XGB treniran.

🪡 Stacking (meta-Ridge) ...
   meta-Ridge fitovan na poslednjih 884 trening uzoraka.

🧪 Back-test (poslednjih 100 izvlačenja):
   model       hits/7    hit%     AUC    LRAP
   DTR          1.490   21.3%   0.506   0.253
   KNN          1.380   19.7%   0.521   0.265
   RFR          1.150   16.4%   0.492   0.250
   LR           1.220   17.4%   0.519   0.242
   GBR          1.190   17.0%   0.504   0.249
   XGB          1.130   16.1%   0.508   0.243
   ENSEMBLE     1.300   18.6%   0.520   0.250
   STACK        1.400   20.0%   0.522   0.267

   (slučajan baseline ≈ 1.256 hits/7)

🎯 Predikcija SLEDEĆEG kola (po modelu):
   DTR   -> [6, x, 23, y, 26, z, 37]
   KNN   -> [5, x, 13, y, 29, z, 36]
   RFR   -> [7, x, 15, y, 27, z, 38]
   LR    -> [7, x, 19, y, 25, z, 37]
   GBR   -> [9, x, 21, y, 25, z, 35]
   XGB   -> [13, x, 21, y, 27, z, 30]

🏁 ANSAMBL (prosek skorova): [7, x, 23, y, 30, z, 35]
🏁 STACKING (meta-Ridge)  : [5, x, 13, y, 23, z, 36]

✅ ANSAMBL validan (suma=165, neparnih=5/7, niskih(≤19)=2/7, raspon=28).
✅ STACKING validan (suma=135, neparnih=3/7, niskih(≤19)=4/7, raspon=31).
📝 Snimljeno u: /Users/4c/Desktop/GHQ/KvantniRegresor/KlasicniRegresori_v2_predikcija.txt

🔁 KlasicniRegresori_v2 — stop  24.05.2026_10.02.16

"""
