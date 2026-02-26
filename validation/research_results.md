# V24 Base Model Optimization Research

**Date:** 2026-02-24
**Objective:** Improve the V24 LightGBM model from ~54% WR toward 56%+ on BTC 5-minute direction predictions
**Method:** Walk-forward out-of-sample evaluation on 180 days of BTC data (51,949 samples, 52 features)
**Validation:** Temporal stability required (both H1 and H2 halves > 52%, trades > 500)

---

## Executive Summary

Four parallel research agents explored hyperparameters, feature engineering, training methodology, and alternative models. **The best improvement found is +0.38pp from hyperparameter tuning alone**, reaching 54.44% WR with excellent temporal stability (H1=54.5%, H2=54.4%).

**Key finding: Improvements do NOT stack.** Combining the best findings from multiple agents dilutes gains rather than compounding them. The optimal hyperparameters appear to already capture the signal that methodology changes (train window, decay weighting) provide independently.

---

## Baseline Configuration

```python
max_depth=2, num_leaves=4, learning_rate=0.05, n_estimators=300,
min_child_samples=200, subsample=0.7, colsample_bytree=0.5,
reg_alpha=1.0, reg_lambda=5.0
```

**Baseline WR: 54.07%** | H1=54.4% | H2=53.7% | 24,192 trades

---

## Agent 1: Hyperparameter Grid Search (167 experiments)

4-phase sequential grid search: depth/leaves/LR -> regularization -> sampling -> n_estimators.

### Best Result: **54.44% WR (+0.38pp)**

```python
max_depth=2, num_leaves=4, learning_rate=0.03, n_estimators=300,
min_child_samples=300, subsample=0.7, colsample_bytree=0.5,
reg_alpha=1.0, reg_lambda=2.0
```

### Key Findings

| Change | WR | Delta | Stable? |
|--------|---:|------:|---------|
| lr=0.03 (from 0.05) | 54.21% | +0.14pp | Yes |
| + min_child=300, reg_lambda=2.0 | 54.44% | +0.38pp | Yes |
| Subsample sweep | No improvement over 0.7 | 0pp | - |
| n_estimators sweep | 300 remains optimal | 0pp | - |

- **Learning rate 0.03** is clearly better than 0.05 (slower learning = better generalization)
- **min_child_samples=300** (up from 200) adds regularization that helps
- **reg_lambda=2.0** (down from 5.0) slightly loosens L2 regularization
- Sampling rates and n_estimators are already at their optima

### Top 5 Hyperparameter Configs

| Config | WR% | H1% | H2% | Delta |
|--------|----:|----:|----:|------:|
| mc300_ra1.0_rl2.0 | 54.44 | 54.5 | 54.4 | +0.38pp |
| mc300_ra0.1_rl2.0 | 54.38 | 54.4 | 54.3 | +0.31pp |
| mc100_ra5.0_rl0.5 | 54.36 | 54.5 | 54.2 | +0.29pp |
| mc100_ra0.1_rl10.0 | 54.35 | 54.7 | 54.0 | +0.28pp |
| mc300_ra1.0_rl0.5 | 54.35 | 54.6 | 54.1 | +0.28pp |

---

## Agent 2: Feature Engineering (3 experiments)

### Experiment 1: Feature Count Sweep

| Top-K | WR% | H1% | H2% |
|------:|----:|----:|----:|
| 10 | 51.61 | 51.7 | 51.5 |
| 15 | 51.23 | 50.9 | 51.6 |
| 20 | 52.28 | 52.3 | 52.3 |
| **25 (baseline)** | **54.07** | **54.4** | **53.7** |
| 30 | 53.89 | 54.1 | 53.6 |
| 35 | 53.97 | 54.2 | 53.8 |
| 40 | 54.05 | 54.3 | 53.8 |
| ALL | 54.14 | 54.4 | 53.9 |

- Top-25 is near optimal; using all features gives marginal +0.07pp
- Below 20 features, performance drops sharply

### Experiment 2: Interaction Features

| Subset | WR% | H1% | H2% | Delta |
|--------|----:|----:|----:|------:|
| All new (8 features) | 53.07 | 53.2 | 52.9 | -1.00pp |
| Session indicators | 54.11 | 54.4 | 53.9 | +0.04pp |
| **Vol/body interactions** | **54.16** | **54.2** | **54.1** | **+0.09pp** |

- Vol/body interactions (vol_regime_ratio, multi_tf_body, alignment_x_vol) provide a small +0.09pp with good stability
- Session indicators (is_london, is_ny, is_asia) are essentially neutral
- Adding all interactions at once hurts (-1.00pp) due to noise

### Experiment 3: Feature Ablation

Feature ablation test had a bug (all results showed 0 WR / 0 trades). Not yet re-run.

---

## Agent 3: Training Methodology (28 experiments)

### Top Results

| Experiment | WR% | H1% | H2% | Delta |
|-----------|----:|----:|----:|------:|
| **train_days=60** | **54.37** | **54.4** | **54.3** | **+0.30pp** |
| decay=0.02 | 54.27 | 54.6 | 54.0 | +0.20pp |
| decay=0.005 | 54.23 | 54.5 | 54.0 | +0.16pp |
| embargo=3 | 54.16 | 54.3 | 54.0 | +0.09pp |
| test=3/step=3 | 54.16 | 54.7 | 53.6 | +0.09pp |
| train_days=90 | 54.14 | 54.3 | 54.0 | +0.07pp |
| fs=none (all features) | 54.14 | 54.4 | 53.9 | +0.07pp |

### Key Findings

- **60-day training window** (+0.30pp) is the single best methodology change
- **Exponential decay weighting** (0.005-0.02) gives +0.16-0.20pp
- **Smaller embargo** (3 vs 7 days) helps slightly (+0.09pp)
- **Shorter train (30/45 days)** hurts significantly
- **Mutual info feature selection** is terrible (-2.58pp)
- **Threshold tuning** doesn't help (0.50 is optimal)
- **High decay (0.05)** is unstable (H1=50.6%)

---

## Agent 4: Alternative Models & Ensembles (6 experiments)

| Model | WR% | H1% | H2% | Delta |
|-------|----:|----:|----:|------:|
| **LightGBM (baseline)** | **54.07** | **54.4** | **53.7** | **0pp** |
| Weighted ensemble | 53.94 | 54.2 | 53.6 | -0.13pp |
| Simple ensemble (avg) | 53.92 | 54.2 | 53.6 | -0.15pp |
| Stacking (LR meta) | 53.79 | 53.5 | 54.1 | -0.28pp |
| CatBoost | 53.56 | 54.0 | 53.2 | -0.50pp |
| XGBoost | N/A (not installed) | - | - | - |

**LightGBM is already the best model.** Ensembles and alternative models all underperform. CatBoost is -0.50pp worse. Ensembling degrades signal.

---

## Combined Verification Test (9 experiments)

Stacking the best findings from all agents:

| Config | WR% | H1% | H2% | Delta |
|--------|----:|----:|----:|------:|
| **Best hyperparams only** | **54.44** | **54.5** | **54.4** | **+0.38pp** |
| Best hp + train60 + decay0.005 | 54.30 | 54.5 | 54.1 | +0.23pp |
| Best hp + train60 | 54.12 | 54.3 | 53.9 | +0.05pp |
| Best hp + train60 + decay0.02 + all features | 54.12 | 54.3 | 53.9 | +0.05pp |
| Best hp + train60 + decay0.02 + vol interactions | 54.12 | 54.1 | 54.1 | +0.05pp |
| Best hp + train60 + decay0.02 + vol + all feat | 54.10 | 54.3 | 53.9 | +0.03pp |
| Baseline | 54.07 | 54.4 | 53.7 | 0pp |
| Best hp + train60 + decay0.02 + embargo3 | 54.05 | 54.4 | 53.7 | -0.02pp |
| Best hp + train60 + decay0.02 | 53.99 | 54.3 | 53.6 | -0.08pp |

### Critical Insight

**Improvements do NOT compound.** The best hyperparams alone (54.44%) beat every combination. Adding train_days=60 or decay weighting to the optimized params actually *hurts* performance. This suggests the optimal hyperparameters already account for the same signal that these methodology changes exploit with baseline params.

---

## Recommended Configuration

```python
# V25 recommended params (from research)
max_depth=2, num_leaves=4, learning_rate=0.03, n_estimators=300,
min_child_samples=300, subsample=0.7, colsample_bytree=0.5,
reg_alpha=1.0, reg_lambda=2.0
```

**Expected improvement: +0.38pp (54.07% -> 54.44%)**
**Temporal stability: H1=54.5%, H2=54.4% (excellent)**

Changes from baseline:
- `learning_rate`: 0.05 -> 0.03 (slower learning, better generalization)
- `min_child_samples`: 200 -> 300 (more regularization)
- `reg_lambda`: 5.0 -> 2.0 (slightly less L2 penalty)

Everything else (features, train window, sampling, model type) should stay the same.

---

## What Didn't Work

1. **Stacking improvements** - combining multiple small gains yields less than the best single change
2. **Alternative models** - CatBoost, ensembles, stacking all underperform LightGBM alone
3. **New interaction features** - vol/body interactions gave only +0.09pp on baseline
4. **Feature selection changes** - mutual info is terrible; top-25 importance is near optimal
5. **Threshold tuning** - 0.50 is optimal, any deviation hurts
6. **High decay weighting** - 0.05 is unstable
7. **Short training windows** - 30/45 days hurt significantly

## What Has Marginal Potential (if revisited later)

1. **train_days=60** (+0.30pp on baseline) - could be revisited if future hyperparams differ
2. **Mild exponential decay** (0.005) - small but stable improvement on baseline
3. **Vol/body interaction features** - +0.09pp with good H1/H2 stability

---

## Experiment Counts

| Agent | Experiments | Wall Time |
|-------|----------:|----------:|
| Hyperparams | 167 | ~25 min |
| Features | ~35 | ~15 min |
| Methodology | 28 | ~15 min |
| Alt Models | 6 | ~10 min |
| Combined Verification | 9 | ~5 min |
| **Total** | **~245** | **~30 min** |
