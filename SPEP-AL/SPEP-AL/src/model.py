
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

import xgboost

DEFAULT_ENSEMBLE_SIZE = 16
DEFAULT_BOOTSTRAP_SAMPLE_FRAC = 1.0
DEFAULT_UCB_BETA = 0.5

BOOTSTRAP_SEED_STRIDE = 7919

TINY_SAMPLE_FALLBACK_REGRESSOR = "ridge"

DEFAULT_ENABLE_TAIL_WEIGHTING = True
DEFAULT_TAIL_WEIGHT_MODE = "quantile_linear"
DEFAULT_TAIL_WEIGHT_QUANTILE = 0.80
DEFAULT_TAIL_WEIGHT_STRENGTH = 2.0
DEFAULT_TAIL_WEIGHT_MAX = 5.0

DEFAULT_ENABLE_HEAD_RERANK = True
DEFAULT_HEAD_RERANK_POOL_SIZE = 64
DEFAULT_HEAD_RERANK_WEIGHT = 0.15
DEFAULT_HEAD_RERANK_TAIL_QUANTILE = 0.80

DEFAULT_ENABLE_SUCCESS_HEAD = False
DEFAULT_SUCCESS_LABEL_MODE = "activity_binary"
DEFAULT_SUCCESS_THRESHOLD = 0.50
DEFAULT_SUCCESS_POSITIVE_QUANTILE = 0.70
DEFAULT_SUCCESS_CLASSIFIER_N_ESTIMATORS = 300

DEFAULT_SUCCESS_BLEND_CLS_WEIGHT_FULL = 0.55
DEFAULT_SUCCESS_BLEND_CLS_WEIGHT_LOW_N = 0.40
DEFAULT_SUCCESS_REG_STD_EPS = 1e-8
DEFAULT_SUCCESS_SCORE_RISK_PENALTY = 0.10

warnings.filterwarnings("ignore", category=UserWarning)


KNOWN_STRUCT_FEATURES = {
    "mean_plddt",
    "local_plddt",
    "local_sasa",
    "rg",
    "dist_to_center",
    "contact_count",
    "neighbor_mean_plddt",
    "delta_vol",
    "delta_hydro",
    "delta_charge",
    "global_mean_plddt",
    "mutation_plddt",
    "window5_mean_plddt",
    "window7_mean_plddt",
    "mutation_sasa",
    "window5_mean_sasa",
    "relative_sasa",
    "position_norm",
    "contact_count_6A",
    "contact_count_8A",
    "contact_count_10A",
    "neighbor10_mean_plddt",
    "neighbor10_mean_sasa",
    "neighbor_hydrophobic_mean",
    "neighbor_charge_mean",
    "is_helix",
    "is_sheet",
    "is_loop",
    "ss_helix",
    "ss_sheet",
    "ss_loop",
    "buried_flag",
    "exposed_flag",
    "ca_to_centroid",
    "wt_local_plddt",
    "mut_local_plddt",
    "delta_local_plddt",
    "wt_local_sasa",
    "mut_local_sasa",
    "delta_local_sasa",
    "delta_contact_count",
    "local_ca_rmsd",
    "wt_mismatch_flag",
    "has_struct",
    "wt_site_relative_sasa",
    "wt_contact_count_8A",
    "wt_local_mean_relative_sasa",
    "wt_local_mean_ca_distance",
    "wt_local_std_ca_distance",
    "wt_local_radius_of_gyration",
    "wt_site_to_centroid_normalized",
    "wt_sin_phi",
    "wt_cos_phi",
    "wt_sin_psi",
    "wt_cos_psi",
    "delta_site_relative_sasa",
    "delta_local_mean_relative_sasa",
    "delta_contact_count_8A",
    "mutation_site_ca_displacement",
    "local_distance_rmsd",
    "local_max_ca_displacement",
    "delta_local_radius_of_gyration",
}


def _safe_sample_without_replacement(values, n, random_seed=None):
    values = np.array(list(values))
    if len(values) == 0:
        return np.array([], dtype=object)
    n = min(int(n), len(values))
    rng = np.random.default_rng(random_seed)
    return rng.choice(values, size=n, replace=False)


def _is_int_like(col):
    if isinstance(col, (int, np.integer)):
        return True
    s = str(col)
    return re.fullmatch(r"-?\d+", s) is not None


def _deduplicate_columns_preserve_prefix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen = {}
    new_cols = []

    for col in df.columns:
        col_str = str(col)
        if col_str not in seen:
            seen[col_str] = 0
            new_cols.append(col_str)
        else:
            seen[col_str] += 1
            new_cols.append(f"{col_str}__dup{seen[col_str]}")

    df.columns = new_cols
    return df


def _split_feature_blocks(df: pd.DataFrame):
    columns = list(df.columns)
    col_strs = [str(c) for c in columns]

    has_prefixed_plm = any(s.startswith("plm__") for s in col_strs)
    has_prefixed_struct = any(s.startswith("struct__") for s in col_strs)

    if has_prefixed_plm or has_prefixed_struct:
        plm_cols = [c for c in columns if str(c).startswith("plm__")]
        struct_cols = [c for c in columns if str(c).startswith("struct__")]
        return plm_cols, struct_cols

    explicit_struct_cols = []
    for c in columns:
        s = str(c)
        if s == "has_struct" or s.startswith("struct_missing__") or s in KNOWN_STRUCT_FEATURES:
            explicit_struct_cols.append(c)

    explicit_struct_cols = list(dict.fromkeys(explicit_struct_cols))

    struct_cols = list(explicit_struct_cols)
    if len(explicit_struct_cols) > 0:
        for c in columns:
            if c in struct_cols:
                continue
            if not _is_int_like(c):
                struct_cols.append(c)

    plm_cols = [c for c in columns if c not in struct_cols]
    return plm_cols, struct_cols


def _rename_block_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}__{i}" for i in range(df.shape[1])]
    return df


def _make_full_feature_matrix_unique(df: pd.DataFrame) -> pd.DataFrame:
    return _deduplicate_columns_preserve_prefix(df)


def _safe_fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.shape[1] == 0:
        return df
    medians = df.median(axis=0, skipna=True).fillna(0.0)
    return df.fillna(medians)


def _safe_spearman(y_true, y_pred):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    corr, _ = spearmanr(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
    return corr


def _safe_r2(y_true, y_pred):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    return r2_score(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])


def _safe_mse(y_true, y_pred):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if np.sum(mask) < 1:
        return np.nan
    return mean_squared_error(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])


def _safe_min_pairwise_distance(X_from: pd.DataFrame, X_to: pd.DataFrame):
    if X_from is None:
        return np.array([])
    if len(X_from) == 0:
        return np.array([])
    if X_to is None or len(X_to) == 0:
        return np.full(len(X_from), np.nan)
    return cdist(X_from, X_to, metric="euclidean").min(axis=1)


def _safe_min_pairwise_distance_excluding_self(X: pd.DataFrame):
    if X is None or len(X) == 0:
        return np.array([])
    if len(X) == 1:
        return np.array([np.nan])

    dmat = cdist(X, X, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    return dmat.min(axis=1)


def _rf_prediction_std(model: RandomForestRegressor, X: pd.DataFrame):
    if model is None or not hasattr(model, "estimators_") or len(X) == 0:
        return np.zeros(len(X))
    preds = np.vstack([tree.predict(X) for tree in model.estimators_])
    return preds.std(axis=0)


def _safe_alpha_from_model(model):
    if model is None:
        return 0
    alpha = getattr(model, "alpha_", 0)
    if isinstance(alpha, (list, tuple, np.ndarray)):
        try:
            return float(np.mean(np.asarray(alpha, dtype=float)))
        except Exception:
            return 0
    try:
        return float(alpha)
    except Exception:
        return 0


def _resolve_regression_type_for_small_n(regression_type: str, n_train_samples: int) -> str:
    if n_train_samples is None:
        return regression_type

    if n_train_samples < 2:
        return TINY_SAMPLE_FALLBACK_REGRESSOR

    if regression_type in {"lasso", "elasticnet", "neuralnet", "gradientboosting", "gp"} and n_train_samples < 5:
        return TINY_SAMPLE_FALLBACK_REGRESSOR

    if regression_type == "knn" and n_train_samples < 3:
        return TINY_SAMPLE_FALLBACK_REGRESSOR

    return regression_type


def _safe_fit_with_optional_sample_weight(model, X_train, y_train, sample_weights=None):
    if sample_weights is None:
        model.fit(X_train, y_train)
        return model

    sw = np.asarray(sample_weights, dtype=float)
    if len(sw) != len(y_train):
        model.fit(X_train, y_train)
        return model

    try:
        model.fit(X_train, y_train, sample_weight=sw)
    except TypeError:
        model.fit(X_train, y_train)
    except ValueError:
        model.fit(X_train, y_train)

    return model


def _safe_numeric_array(values, fill_value=np.nan):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isnan(fill_value):
        return arr
    arr = np.where(np.isfinite(arr), arr, fill_value)
    return arr


def _rank_normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    s = pd.Series(values)
    valid = s.notna()
    out = pd.Series(np.zeros(len(s), dtype=float))
    if valid.sum() == 0:
        return out.to_numpy(dtype=float)
    out.loc[valid] = s.loc[valid].rank(method="average", pct=True).astype(float)
    return out.to_numpy(dtype=float)


def _make_tail_reference(y_train: np.ndarray, y_train_activity_scaled: Optional[np.ndarray] = None) -> np.ndarray:
    if y_train_activity_scaled is not None:
        ref = _safe_numeric_array(y_train_activity_scaled, fill_value=np.nan)
        if np.isfinite(ref).sum() >= 2:
            return ref

    raw = _safe_numeric_array(y_train, fill_value=np.nan)
    return _rank_normalize_array(raw)


def _compute_tail_sample_weights(
    y_train: np.ndarray,
    y_train_activity_scaled: Optional[np.ndarray] = None,
    *,
    enable_tail_weighting: bool = DEFAULT_ENABLE_TAIL_WEIGHTING,
    tail_weight_mode: str = DEFAULT_TAIL_WEIGHT_MODE,
    tail_weight_quantile: float = DEFAULT_TAIL_WEIGHT_QUANTILE,
    tail_weight_strength: float = DEFAULT_TAIL_WEIGHT_STRENGTH,
    tail_weight_max: float = DEFAULT_TAIL_WEIGHT_MAX,
) -> np.ndarray:
    n = len(y_train)
    if n == 0:
        return np.array([], dtype=float)

    if not enable_tail_weighting:
        return np.ones(n, dtype=float)

    ref = _make_tail_reference(y_train, y_train_activity_scaled=y_train_activity_scaled)
    valid = np.isfinite(ref)

    if valid.sum() < 3:
        return np.ones(n, dtype=float)

    q = float(min(0.99, max(0.50, tail_weight_quantile)))
    threshold = float(np.nanquantile(ref[valid], q))

    weights = np.ones(n, dtype=float)
    strength = float(max(0.0, tail_weight_strength))
    max_weight = float(max(1.0, tail_weight_max))

    mode = str(tail_weight_mode).strip().lower()

    if mode == "quantile_binary":
        weights[valid & (ref >= threshold)] += strength

    else:
        tail_excess = np.zeros(n, dtype=float)
        tail_mask = valid & (ref >= threshold)

        denom = float(np.nanmax(ref[valid]) - threshold)
        if denom < 1e-12:
            tail_excess[tail_mask] = 1.0
        else:
            tail_excess[tail_mask] = np.clip((ref[tail_mask] - threshold) / denom, 0.0, 1.0)

        weights += strength * tail_excess

    weights = np.clip(weights, 1.0, max_weight)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return weights.astype(float)


def _bootstrap_sample_indices(
    n_samples: int,
    sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    random_seed: int = 1,
    sample_weights: Optional[np.ndarray] = None,
):
    rng = np.random.default_rng(random_seed)
    if n_samples <= 0:
        return np.array([], dtype=int)

    sample_size = max(1, int(round(n_samples * sample_frac)))
    sample_size = max(sample_size, 2) if n_samples >= 2 else 1
    sample_size = min(sample_size, max(1, n_samples))

    if sample_weights is None:
        p = None
    else:
        p = np.asarray(sample_weights, dtype=float).reshape(-1)
        if len(p) != n_samples:
            p = None
        else:
            p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
            if p.sum() <= 0:
                p = None
            else:
                p = p / p.sum()

    return rng.choice(n_samples, size=sample_size, replace=True, p=p)


def _safe_standardize_train_test_for_distance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    X_train_num = _safe_fill_numeric(X_train)
    X_test_num = _safe_fill_numeric(X_test)

    train_arr = X_train_num.to_numpy(dtype=float)
    test_arr = X_test_num.to_numpy(dtype=float)

    if train_arr.shape[1] == 0:
        return train_arr, test_arr

    mu = np.nanmean(train_arr, axis=0)
    sd = np.nanstd(train_arr, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    train_std = (train_arr - mu) / sd
    test_std = (test_arr - mu) / sd

    train_std = np.where(np.isfinite(train_std), train_std, 0.0)
    test_std = np.where(np.isfinite(test_std), test_std, 0.0)
    return train_std, test_std


def _compute_head_rerank_scores(
    X_train: pd.DataFrame,
    y_train_ref: np.ndarray,
    X_query: pd.DataFrame,
    *,
    tail_quantile: float = DEFAULT_HEAD_RERANK_TAIL_QUANTILE,
) -> np.ndarray:
    if X_query is None or len(X_query) == 0:
        return np.array([], dtype=float)

    if X_train is None or len(X_train) < 6:
        return np.zeros(len(X_query), dtype=float)

    y_ref = _safe_numeric_array(y_train_ref, fill_value=np.nan)
    valid = np.isfinite(y_ref)
    if valid.sum() < 6:
        return np.zeros(len(X_query), dtype=float)

    q = float(min(0.99, max(0.50, tail_quantile)))
    threshold = float(np.nanquantile(y_ref[valid], q))

    tail_mask = valid & (y_ref >= threshold)
    non_tail_mask = valid & (y_ref < threshold)

    if tail_mask.sum() < 2 or non_tail_mask.sum() < 2:
        return np.zeros(len(X_query), dtype=float)

    train_std, query_std = _safe_standardize_train_test_for_distance(X_train, X_query)

    tail_ref = train_std[tail_mask]
    non_tail_ref = train_std[non_tail_mask]

    if len(tail_ref) == 0 or len(non_tail_ref) == 0:
        return np.zeros(len(X_query), dtype=float)

    dist_to_tail = cdist(query_std, tail_ref, metric="euclidean").mean(axis=1)
    dist_to_non_tail = cdist(query_std, non_tail_ref, metric="euclidean").mean(axis=1)

    score_raw = dist_to_non_tail - dist_to_tail
    score_raw = np.where(np.isfinite(score_raw), score_raw, 0.0)
    return score_raw.astype(float)


def _apply_head_rerank_to_test_predictions(
    base_pred_test: np.ndarray,
    X_train_rerank: pd.DataFrame,
    y_train_ref: np.ndarray,
    X_test_rerank: pd.DataFrame,
    *,
    enable_head_rerank: bool = DEFAULT_ENABLE_HEAD_RERANK,
    head_rerank_pool_size: int = DEFAULT_HEAD_RERANK_POOL_SIZE,
    head_rerank_weight: float = DEFAULT_HEAD_RERANK_WEIGHT,
    head_rerank_tail_quantile: float = DEFAULT_HEAD_RERANK_TAIL_QUANTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_pred_test = np.asarray(base_pred_test, dtype=float)

    n_test = len(base_pred_test)
    if n_test == 0:
        return (
            base_pred_test.copy(),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=bool),
        )

    final_pred_test = base_pred_test.copy()
    rerank_score_all = np.zeros(n_test, dtype=float)
    rerank_correction_all = np.zeros(n_test, dtype=float)
    rerank_applied_mask = np.zeros(n_test, dtype=bool)

    if not enable_head_rerank:
        return final_pred_test, rerank_score_all, rerank_correction_all, rerank_applied_mask

    if X_train_rerank is None or len(X_train_rerank) < 6 or X_test_rerank is None or len(X_test_rerank) == 0:
        return final_pred_test, rerank_score_all, rerank_correction_all, rerank_applied_mask

    pool_k = int(max(1, head_rerank_pool_size))
    pool_k = min(pool_k, n_test)
    order = np.argsort(-base_pred_test)
    head_idx = order[:pool_k]

    X_query = X_test_rerank.iloc[head_idx, :].copy()
    score_raw = _compute_head_rerank_scores(
        X_train=X_train_rerank,
        y_train_ref=y_train_ref,
        X_query=X_query,
        tail_quantile=head_rerank_tail_quantile,
    )

    if len(score_raw) == 0:
        return final_pred_test, rerank_score_all, rerank_correction_all, rerank_applied_mask

    score_raw = np.asarray(score_raw, dtype=float)
    score_raw = np.where(np.isfinite(score_raw), score_raw, 0.0)

    score_rank = _rank_normalize_array(score_raw)
    score_rank = np.where(np.isfinite(score_rank), score_rank, 0.0)

    centered_score = score_rank - float(np.nanmean(score_rank))
    max_abs = float(np.nanmax(np.abs(centered_score))) if len(centered_score) > 0 else 0.0
    if not np.isfinite(max_abs) or max_abs < 1e-12:
        normalized_score = np.zeros_like(centered_score, dtype=float)
    else:
        normalized_score = centered_score / max_abs

    normalized_score = np.clip(
        np.where(np.isfinite(normalized_score), normalized_score, 0.0),
        -1.0,
        1.0,
    )

    weight = float(max(0.0, head_rerank_weight))
    correction = weight * normalized_score
    correction = np.clip(correction, -weight, weight)

    rerank_score_all[head_idx] = normalized_score
    rerank_correction_all[head_idx] = correction
    rerank_applied_mask[head_idx] = True
    final_pred_test[head_idx] = final_pred_test[head_idx] + correction

    return final_pred_test, rerank_score_all, rerank_correction_all, rerank_applied_mask


def _build_single_model(regression_type: str, n_train_samples: Optional[int] = None, random_seed: int = 1):
    regression_type = _resolve_regression_type_for_small_n(regression_type, n_train_samples or 0)

    if regression_type == "ridge":
        return linear_model.RidgeCV(alphas=np.logspace(-6, 6, 25))

    elif regression_type == "lasso":
        cv = max(2, min(5, int(n_train_samples or 5)))
        return linear_model.LassoCV(
            cv=cv,
            max_iter=100000,
            tol=1e-3,
            random_state=random_seed,
        )

    elif regression_type == "elasticnet":
        cv = max(2, min(5, int(n_train_samples or 5)))
        return linear_model.ElasticNetCV(
            cv=cv,
            max_iter=100000,
            tol=1e-3,
            random_state=random_seed,
        )

    elif regression_type == "linear":
        return linear_model.LinearRegression()

    elif regression_type == "neuralnet":
        return MLPRegressor(
            hidden_layer_sizes=(64, 16),
            max_iter=2000,
            random_state=random_seed,
            early_stopping=True,
        )

    elif regression_type == "randomforest":
        return RandomForestRegressor(
            n_estimators=300,
            criterion="friedman_mse",
            random_state=random_seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        )

    elif regression_type == "gradientboosting":
        return xgboost.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.8,
            subsample=0.8,
            learning_rate=0.05,
            max_depth=5,
            reg_alpha=1.0,
            n_estimators=300,
            random_state=random_seed,
            verbosity=0,
        )

    elif regression_type == "knn":
        n_neighbors = max(1, min(5, int(n_train_samples or 5)))
        return KNeighborsRegressor(n_neighbors=n_neighbors)

    elif regression_type == "gp":
        return GaussianProcessRegressor(alpha=1e-6)

    else:
        raise ValueError(f"Unsupported regression_type: {regression_type}")


def _fit_single_model_bundle(
    regression_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_seed: int = 1,
    sample_weights: Optional[np.ndarray] = None,
):
    model = _build_single_model(
        regression_type=regression_type,
        n_train_samples=len(y_train),
        random_seed=random_seed,
    )
    model = _safe_fit_with_optional_sample_weight(
        model=model,
        X_train=X_train,
        y_train=y_train,
        sample_weights=sample_weights,
    )
    alpha = _safe_alpha_from_model(model)
    return model, alpha


def _predict_single_model_bundle(
    model,
    regression_type: str,
    X: pd.DataFrame,
):
    if len(X) == 0:
        return np.array([]), np.array([])

    regression_type = getattr(model, "_resolved_regression_type", regression_type)

    if isinstance(model, GaussianProcessRegressor):
        y_pred, y_std = model.predict(X, return_std=True)
        return np.asarray(y_pred), np.asarray(y_std)

    y_pred = model.predict(X)

    if isinstance(model, RandomForestRegressor):
        y_std = _rf_prediction_std(model, X)
    else:
        y_std = np.zeros(len(y_pred))

    return np.asarray(y_pred), np.asarray(y_std)


def _fit_hybrid_model_bundle(
    X_train_plm: pd.DataFrame,
    X_train_struct: pd.DataFrame,
    y_train: np.ndarray,
    random_seed: int = 1,
    sample_weights: Optional[np.ndarray] = None,
):
    plm_model = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 25))
    plm_model = _safe_fit_with_optional_sample_weight(
        model=plm_model,
        X_train=X_train_plm,
        y_train=y_train,
        sample_weights=sample_weights,
    )

    plm_pred_train = plm_model.predict(X_train_plm)
    residual_target = y_train - plm_pred_train

    struct_model = RandomForestRegressor(
        n_estimators=300,
        criterion="friedman_mse",
        random_state=random_seed,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    struct_model = _safe_fit_with_optional_sample_weight(
        model=struct_model,
        X_train=X_train_struct,
        y_train=residual_target,
        sample_weights=sample_weights,
    )

    bundle = {
        "plm_model": plm_model,
        "struct_model": struct_model,
    }
    alpha = _safe_alpha_from_model(plm_model)
    return bundle, alpha


def _predict_hybrid_model_bundle(
    bundle: Dict[str, Any],
    X_plm: pd.DataFrame,
    X_struct: pd.DataFrame,
):
    if len(X_plm) == 0 and len(X_struct) == 0:
        return np.array([]), np.array([])

    plm_model = bundle["plm_model"]
    struct_model = bundle["struct_model"]

    plm_pred = plm_model.predict(X_plm)
    residual_pred = struct_model.predict(X_struct)

    y_pred = np.asarray(plm_pred) + np.asarray(residual_pred)
    y_std = _rf_prediction_std(struct_model, X_struct)

    return np.asarray(y_pred), np.asarray(y_std)


def _fit_predict_single_model(
    regression_type,
    X_train,
    y_train,
    X_test,
    random_seed: int = 1,
    sample_weights: Optional[np.ndarray] = None,
):
    model, alpha = _fit_single_model_bundle(
        regression_type=regression_type,
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
        sample_weights=sample_weights,
    )

    y_pred_train, y_std_train = _predict_single_model_bundle(model, regression_type, X_train)
    y_pred_test, y_std_test = _predict_single_model_bundle(model, regression_type, X_test)

    return model, y_pred_train, y_pred_test, y_std_train, y_std_test, alpha


def _fit_predict_hybrid_residual(
    X_train_plm,
    X_test_plm,
    X_train_struct,
    X_test_struct,
    y_train,
    random_seed: int = 1,
    sample_weights: Optional[np.ndarray] = None,
):
    bundle, alpha = _fit_hybrid_model_bundle(
        X_train_plm=X_train_plm,
        X_train_struct=X_train_struct,
        y_train=y_train,
        random_seed=random_seed,
        sample_weights=sample_weights,
    )

    y_pred_train, y_std_train = _predict_hybrid_model_bundle(bundle, X_train_plm, X_train_struct)
    y_pred_test, y_std_test = _predict_hybrid_model_bundle(bundle, X_test_plm, X_test_struct)

    return bundle, y_pred_train, y_pred_test, y_std_train, y_std_test, alpha


def _make_success_binary_labels(
    y_train: np.ndarray,
    y_train_activity_scaled: Optional[np.ndarray] = None,
    y_train_activity_binary: Optional[np.ndarray] = None,
    *,
    success_label_mode: str = DEFAULT_SUCCESS_LABEL_MODE,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    success_positive_quantile: float = DEFAULT_SUCCESS_POSITIVE_QUANTILE,
) -> np.ndarray:
    n = len(y_train)
    if n == 0:
        return np.array([], dtype=float)

    mode = str(success_label_mode or "activity_binary").strip().lower()

    def _valid_binary_from(values):
        if values is None:
            return None
        arr = _safe_numeric_array(values, fill_value=np.nan)
        if len(arr) != n or np.isfinite(arr).sum() == 0:
            return None
        out = np.where(arr >= 0.5, 1.0, 0.0)
        out = np.where(np.isfinite(arr), out, np.nan)
        return out

    def _from_scaled_threshold(values):
        if values is None:
            return None
        arr = _safe_numeric_array(values, fill_value=np.nan)
        if len(arr) != n or np.isfinite(arr).sum() == 0:
            return None
        threshold = float(success_threshold)
        out = np.where(arr >= threshold, 1.0, 0.0)
        out = np.where(np.isfinite(arr), out, np.nan)
        return out

    def _from_quantile(values):
        if values is None:
            return None
        arr = _safe_numeric_array(values, fill_value=np.nan)
        valid = np.isfinite(arr)
        if len(arr) != n or valid.sum() < 2:
            return None
        q = float(min(0.99, max(0.01, success_positive_quantile)))
        threshold = float(np.nanquantile(arr[valid], q))
        out = np.where(arr >= threshold, 1.0, 0.0)
        out = np.where(valid, out, np.nan)
        return out

    labels = None

    if mode in {"activity_binary", "binary", "hit", "success"}:
        labels = _valid_binary_from(y_train_activity_binary)
    elif mode in {"scaled_threshold", "threshold", "activity_scaled"}:
        labels = _from_scaled_threshold(y_train_activity_scaled)
    elif mode in {"quantile", "scaled_quantile", "activity_quantile"}:
        labels = _from_quantile(y_train_activity_scaled)
    elif mode == "auto":
        labels = _valid_binary_from(y_train_activity_binary)
        if labels is None or np.isfinite(labels).sum() == 0:
            labels = _from_scaled_threshold(y_train_activity_scaled)
        if labels is None or np.isfinite(labels).sum() == 0:
            labels = _from_quantile(y_train)
    else:
        raise ValueError(f"Unsupported success_label_mode: {success_label_mode}")

    if labels is None:
        labels = _from_quantile(y_train)

    if labels is None:
        labels = np.zeros(n, dtype=float)

    labels = np.asarray(labels, dtype=float)
    valid = np.isfinite(labels)
    if valid.sum() == 0:
        return np.zeros(n, dtype=float)
    fill = 1.0 if float(np.nanmean(labels[valid])) >= 0.5 else 0.0
    labels = np.where(valid, labels, fill)
    labels = np.where(labels >= 0.5, 1.0, 0.0)
    return labels.astype(float)


def _build_success_classifier(random_seed: int = 1, n_train_samples: Optional[int] = None):
    return RandomForestClassifier(
        n_estimators=DEFAULT_SUCCESS_CLASSIFIER_N_ESTIMATORS,
        criterion="gini",
        random_state=random_seed,
        n_jobs=-1,
        min_samples_leaf=1 if (n_train_samples or 0) < 8 else 2,
        max_features="sqrt",
        class_weight="balanced_subsample",
    )


def _safe_fit_success_classifier(model, X_train, y_success, sample_weights=None):
    y_success = np.asarray(y_success, dtype=int)
    if sample_weights is None:
        model.fit(X_train, y_success)
        return model

    sw = np.asarray(sample_weights, dtype=float)
    if len(sw) != len(y_success):
        model.fit(X_train, y_success)
        return model

    try:
        model.fit(X_train, y_success, sample_weight=sw)
    except TypeError:
        model.fit(X_train, y_success)
    except ValueError:
        model.fit(X_train, y_success)
    return model


def _predict_success_probability(model, X: pd.DataFrame, fallback_prob: float = 0.5) -> np.ndarray:
    if X is None or len(X) == 0:
        return np.array([], dtype=float)

    if model is None or not hasattr(model, "predict_proba"):
        return np.full(len(X), float(fallback_prob), dtype=float)

    try:
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is None:
            return np.full(len(X), float(fallback_prob), dtype=float)
        classes = np.asarray(classes)
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            out = proba[:, idx]
        else:
            out = np.zeros(len(X), dtype=float)
        out = np.asarray(out, dtype=float)
        out = np.where(np.isfinite(out), out, float(fallback_prob))
        return np.clip(out, 0.0, 1.0)
    except Exception:
        return np.full(len(X), float(fallback_prob), dtype=float)


def _fit_predict_success_bootstrap_ensemble(
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_success_train: np.ndarray,
    *,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    random_seed: int = 1,
):
    y_success_train = np.asarray(y_success_train, dtype=float)
    n_train = len(y_success_train)

    if n_train == 0:
        raise ValueError("Training set is empty in success bootstrap ensemble.")

    valid = np.isfinite(y_success_train)
    if valid.sum() == 0:
        base_prob = 0.5
        y_clean = np.zeros(n_train, dtype=int)
    else:
        y_clean = np.where(y_success_train >= 0.5, 1, 0).astype(int)
        base_prob = float(np.mean(y_clean[valid])) if valid.sum() > 0 else 0.5
        base_prob = float(np.clip(base_prob, 0.0, 1.0))

    unique_classes = np.unique(y_clean)
    if len(unique_classes) < 2 or n_train < 3:
        prob_train = np.full(n_train, base_prob, dtype=float)
        prob_test = np.full(len(X_test_full), base_prob, dtype=float)
        std_train = np.zeros(n_train, dtype=float)
        std_test = np.zeros(len(X_test_full), dtype=float)
        model_bundle = {
            "type": "success_head_constant",
            "members": [],
            "ensemble_size": 0,
            "base_prob": base_prob,
            "reason": "single_class_or_too_few_samples",
        }
        return model_bundle, prob_train, prob_test, std_train, std_test

    if ensemble_size is None or ensemble_size <= 1:
        clf = _build_success_classifier(random_seed=random_seed, n_train_samples=n_train)
        clf = _safe_fit_success_classifier(clf, X_train_full, y_clean, sample_weights=None)
        prob_train = _predict_success_probability(clf, X_train_full, fallback_prob=base_prob)
        prob_test = _predict_success_probability(clf, X_test_full, fallback_prob=base_prob)
        std_train = np.zeros(n_train, dtype=float)
        std_test = np.zeros(len(X_test_full), dtype=float)
        model_bundle = {
            "type": "success_head_single_classifier",
            "members": [clf],
            "ensemble_size": 1,
            "base_prob": base_prob,
        }
        return model_bundle, prob_train, prob_test, std_train, std_test

    train_member_probs = []
    test_member_probs = []
    member_models = []

    for b in range(int(ensemble_size)):
        member_seed = int(random_seed + BOOTSTRAP_SEED_STRIDE * (b + 1) + 104729)
        boot_idx = _bootstrap_sample_indices(
            n_samples=n_train,
            sample_frac=bootstrap_sample_frac,
            random_seed=member_seed,
            sample_weights=None,
        )

        y_boot = y_clean[boot_idx]
        X_boot = X_train_full.iloc[boot_idx, :].reset_index(drop=True)

        if len(np.unique(y_boot)) < 2:
            member_prob = float(np.mean(y_boot)) if len(y_boot) > 0 else base_prob
            pred_train_b = np.full(n_train, member_prob, dtype=float)
            pred_test_b = np.full(len(X_test_full), member_prob, dtype=float)
            member_models.append({"type": "constant", "prob": member_prob})
        else:
            clf = _build_success_classifier(random_seed=member_seed, n_train_samples=len(y_boot))
            clf = _safe_fit_success_classifier(clf, X_boot, y_boot, sample_weights=None)
            pred_train_b = _predict_success_probability(clf, X_train_full, fallback_prob=base_prob)
            pred_test_b = _predict_success_probability(clf, X_test_full, fallback_prob=base_prob)
            member_models.append(clf)

        train_member_probs.append(np.asarray(pred_train_b, dtype=float))
        test_member_probs.append(np.asarray(pred_test_b, dtype=float))

    train_member_probs = np.vstack(train_member_probs)
    test_member_probs = np.vstack(test_member_probs)

    prob_mean_train = train_member_probs.mean(axis=0)
    prob_mean_test = test_member_probs.mean(axis=0)
    prob_std_train = train_member_probs.std(axis=0, ddof=1)
    prob_std_test = test_member_probs.std(axis=0, ddof=1)

    prob_mean_train = np.clip(np.where(np.isfinite(prob_mean_train), prob_mean_train, base_prob), 0.0, 1.0)
    prob_mean_test = np.clip(np.where(np.isfinite(prob_mean_test), prob_mean_test, base_prob), 0.0, 1.0)
    prob_std_train = np.where(np.isfinite(prob_std_train), prob_std_train, 0.0)
    prob_std_test = np.where(np.isfinite(prob_std_test), prob_std_test, 0.0)

    model_bundle = {
        "type": "success_head_bootstrap_ensemble",
        "members": member_models,
        "ensemble_size": int(ensemble_size),
        "bootstrap_sample_frac": float(bootstrap_sample_frac),
        "base_prob": base_prob,
    }

    return model_bundle, prob_mean_train, prob_mean_test, prob_std_train, prob_std_test


def _infer_success_threshold_in_regression_scale(
    y_train: np.ndarray,
    y_success_train: np.ndarray,
    *,
    success_positive_quantile: float = DEFAULT_SUCCESS_POSITIVE_QUANTILE,
) -> float:
    y = _safe_numeric_array(y_train, fill_value=np.nan)
    s = _safe_numeric_array(y_success_train, fill_value=np.nan)

    valid = np.isfinite(y) & np.isfinite(s)
    if valid.sum() < 2:
        finite_y = y[np.isfinite(y)]
        if len(finite_y) == 0:
            return 0.0
        q = float(min(0.99, max(0.01, success_positive_quantile)))
        return float(np.nanquantile(finite_y, q))

    yv = y[valid]
    sv = np.where(s[valid] >= 0.5, 1, 0)
    pos = yv[sv == 1]
    neg = yv[sv == 0]

    if len(pos) > 0 and len(neg) > 0:
        max_neg = float(np.nanmax(neg))
        min_pos = float(np.nanmin(pos))
        if np.isfinite(max_neg) and np.isfinite(min_pos) and max_neg <= min_pos:
            return float(0.5 * (max_neg + min_pos))

        med_neg = float(np.nanmedian(neg))
        med_pos = float(np.nanmedian(pos))
        if np.isfinite(med_neg) and np.isfinite(med_pos):
            return float(0.5 * (med_neg + med_pos))

    q = float(min(0.99, max(0.01, success_positive_quantile)))
    return float(np.nanquantile(yv, q))


def _logistic_success_probability_from_mean_std(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    threshold: float,
) -> np.ndarray:
    mean = np.asarray(pred_mean, dtype=float)
    std = np.asarray(pred_std, dtype=float)
    std = np.where(np.isfinite(std) & (std > DEFAULT_SUCCESS_REG_STD_EPS), std, 1.0)
    z = (mean - float(threshold)) / std
    z = np.clip(np.where(np.isfinite(z), z, 0.0), -40.0, 40.0)
    prob = 1.0 / (1.0 + np.exp(-1.702 * z))
    return np.clip(np.where(np.isfinite(prob), prob, 0.5), 0.0, 1.0)


def _compute_regression_success_probability(
    model_bundle: Any,
    pred_mean_train: np.ndarray,
    pred_mean_test: np.ndarray,
    pred_std_train: np.ndarray,
    pred_std_test: np.ndarray,
    y_train: np.ndarray,
    y_success_train: np.ndarray,
    *,
    success_positive_quantile: float = DEFAULT_SUCCESS_POSITIVE_QUANTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    threshold = _infer_success_threshold_in_regression_scale(
        y_train=y_train,
        y_success_train=y_success_train,
        success_positive_quantile=success_positive_quantile,
    )

    train_member_preds = None
    test_member_preds = None
    if isinstance(model_bundle, dict):
        train_member_preds = model_bundle.get("train_member_preds", None)
        test_member_preds = model_bundle.get("test_member_preds", None)

    if train_member_preds is not None and test_member_preds is not None:
        try:
            train_arr = np.asarray(train_member_preds, dtype=float)
            test_arr = np.asarray(test_member_preds, dtype=float)
            if train_arr.ndim == 2 and test_arr.ndim == 2:
                train_hits = (train_arr >= float(threshold)).astype(float)
                test_hits = (test_arr >= float(threshold)).astype(float)

                prob_train = train_hits.mean(axis=0)
                prob_test = test_hits.mean(axis=0)
                std_train = train_hits.std(axis=0, ddof=1) if train_hits.shape[0] > 1 else np.zeros(train_hits.shape[1])
                std_test = test_hits.std(axis=0, ddof=1) if test_hits.shape[0] > 1 else np.zeros(test_hits.shape[1])

                prob_train = np.clip(np.where(np.isfinite(prob_train), prob_train, 0.5), 0.0, 1.0)
                prob_test = np.clip(np.where(np.isfinite(prob_test), prob_test, 0.5), 0.0, 1.0)
                std_train = np.where(np.isfinite(std_train), std_train, 0.0)
                std_test = np.where(np.isfinite(std_test), std_test, 0.0)
                return prob_train, prob_test, std_train, std_test, float(threshold)
        except Exception:
            pass

    prob_train = _logistic_success_probability_from_mean_std(
        pred_mean=np.asarray(pred_mean_train, dtype=float),
        pred_std=np.asarray(pred_std_train, dtype=float),
        threshold=threshold,
    )
    prob_test = _logistic_success_probability_from_mean_std(
        pred_mean=np.asarray(pred_mean_test, dtype=float),
        pred_std=np.asarray(pred_std_test, dtype=float),
        threshold=threshold,
    )

    std_train = np.sqrt(np.clip(prob_train * (1.0 - prob_train), 0.0, 0.25))
    std_test = np.sqrt(np.clip(prob_test * (1.0 - prob_test), 0.0, 0.25))
    return prob_train, prob_test, std_train, std_test, float(threshold)


def _resolve_success_blend_weight(
    y_success_train: np.ndarray,
    success_model: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int, int, float]:
    y = _safe_numeric_array(y_success_train, fill_value=np.nan)
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return 0.0, 0, 0, np.nan

    y_bin = np.where(y[valid] >= 0.5, 1, 0)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    pos_frac = float(n_pos / max(1, n_pos + n_neg))

    if success_model is not None and isinstance(success_model, dict):
        if success_model.get("type") == "success_head_constant":
            return 0.0, n_pos, n_neg, pos_frac

    if n_pos >= 4 and n_neg >= 4 and (n_pos + n_neg) >= 16:
        return float(DEFAULT_SUCCESS_BLEND_CLS_WEIGHT_FULL), n_pos, n_neg, pos_frac

    if n_pos >= 2 and n_neg >= 2 and (n_pos + n_neg) >= 8:
        return float(DEFAULT_SUCCESS_BLEND_CLS_WEIGHT_LOW_N), n_pos, n_neg, pos_frac

    return 0.0, n_pos, n_neg, pos_frac


def _blend_success_probabilities(
    prob_cls_train: np.ndarray,
    prob_cls_test: np.ndarray,
    std_cls_train: np.ndarray,
    std_cls_test: np.ndarray,
    prob_reg_train: np.ndarray,
    prob_reg_test: np.ndarray,
    std_reg_train: np.ndarray,
    std_reg_test: np.ndarray,
    cls_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = float(min(1.0, max(0.0, cls_weight)))
    wr = 1.0 - w

    prob_train = w * np.asarray(prob_cls_train, dtype=float) + wr * np.asarray(prob_reg_train, dtype=float)
    prob_test = w * np.asarray(prob_cls_test, dtype=float) + wr * np.asarray(prob_reg_test, dtype=float)

    std_train = np.sqrt(
        (w * np.asarray(std_cls_train, dtype=float)) ** 2
        + (wr * np.asarray(std_reg_train, dtype=float)) ** 2
    )
    std_test = np.sqrt(
        (w * np.asarray(std_cls_test, dtype=float)) ** 2
        + (wr * np.asarray(std_reg_test, dtype=float)) ** 2
    )

    prob_train = np.clip(np.where(np.isfinite(prob_train), prob_train, 0.5), 0.0, 1.0)
    prob_test = np.clip(np.where(np.isfinite(prob_test), prob_test, 0.5), 0.0, 1.0)
    std_train = np.where(np.isfinite(std_train), std_train, 0.0)
    std_test = np.where(np.isfinite(std_test), std_test, 0.0)

    return prob_train, prob_test, std_train, std_test


def _fit_predict_bootstrap_ensemble(
    regression_type: str,
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train: np.ndarray,
    *,
    use_hybrid: bool = False,
    X_train_plm: Optional[pd.DataFrame] = None,
    X_test_plm: Optional[pd.DataFrame] = None,
    X_train_struct: Optional[pd.DataFrame] = None,
    X_test_struct: Optional[pd.DataFrame] = None,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    random_seed: int = 1,
    train_sample_weights: Optional[np.ndarray] = None,
):
    n_train = len(y_train)

    if n_train == 0:
        raise ValueError("Training set is empty in bootstrap ensemble.")

    if ensemble_size is None or ensemble_size <= 1:
        if use_hybrid:
            return _fit_predict_hybrid_residual(
                X_train_plm=X_train_plm,
                X_test_plm=X_test_plm,
                X_train_struct=X_train_struct,
                X_test_struct=X_test_struct,
                y_train=y_train,
                random_seed=random_seed,
                sample_weights=train_sample_weights,
            )
        else:
            return _fit_predict_single_model(
                regression_type=regression_type,
                X_train=X_train_full,
                y_train=y_train,
                X_test=X_test_full,
                random_seed=random_seed,
                sample_weights=train_sample_weights,
            )

    train_member_preds = []
    test_member_preds = []
    member_alphas = []
    member_models = []

    for b in range(int(ensemble_size)):
        member_seed = int(random_seed + BOOTSTRAP_SEED_STRIDE * (b + 1))
        boot_idx = _bootstrap_sample_indices(
            n_samples=n_train,
            sample_frac=bootstrap_sample_frac,
            random_seed=member_seed,
            sample_weights=train_sample_weights,
        )

        y_boot = y_train[boot_idx]

        if use_hybrid:
            X_boot_plm = X_train_plm.iloc[boot_idx, :].reset_index(drop=True)
            X_boot_struct = X_train_struct.iloc[boot_idx, :].reset_index(drop=True)

            bundle, alpha = _fit_hybrid_model_bundle(
                X_train_plm=X_boot_plm,
                X_train_struct=X_boot_struct,
                y_train=y_boot,
                random_seed=member_seed,
                sample_weights=None,
            )

            pred_train_b, _ = _predict_hybrid_model_bundle(bundle, X_train_plm, X_train_struct)
            pred_test_b, _ = _predict_hybrid_model_bundle(bundle, X_test_plm, X_test_struct)

        else:
            X_boot_full = X_train_full.iloc[boot_idx, :].reset_index(drop=True)

            bundle, alpha = _fit_single_model_bundle(
                regression_type=regression_type,
                X_train=X_boot_full,
                y_train=y_boot,
                random_seed=member_seed,
                sample_weights=None,
            )

            pred_train_b, _ = _predict_single_model_bundle(bundle, regression_type, X_train_full)
            pred_test_b, _ = _predict_single_model_bundle(bundle, regression_type, X_test_full)

        train_member_preds.append(np.asarray(pred_train_b, dtype=float))
        test_member_preds.append(np.asarray(pred_test_b, dtype=float))
        member_alphas.append(alpha)
        member_models.append(bundle)

    train_member_preds = np.vstack(train_member_preds)
    test_member_preds = np.vstack(test_member_preds)

    pred_mean_train = train_member_preds.mean(axis=0)
    pred_mean_test = test_member_preds.mean(axis=0)

    pred_std_train = train_member_preds.std(axis=0, ddof=1)
    pred_std_test = test_member_preds.std(axis=0, ddof=1)

    alpha_mean = float(np.mean(member_alphas)) if len(member_alphas) > 0 else 0.0

    model_bundle = {
        "type": "bootstrap_ensemble_hybrid" if use_hybrid else "bootstrap_ensemble_single",
        "members": member_models,
        "member_alphas": member_alphas,
        "ensemble_size": int(ensemble_size),
        "bootstrap_sample_frac": float(bootstrap_sample_frac),
        "regression_type": regression_type,
        "use_hybrid": bool(use_hybrid),
        "train_member_preds": train_member_preds,
        "test_member_preds": test_member_preds,
    }

    return (
        model_bundle,
        pred_mean_train,
        pred_mean_test,
        pred_std_train,
        pred_std_test,
        alpha_mean,
    )


def first_round(
    labels,
    embeddings,
    explicit_variants=None,
    num_mutants_per_round=16,
    first_round_strategy="random",
    embedding_type=None,
    random_seed=None,
):
    print("Starting labels length:", len(labels))

    variants_without_WT = labels.variant[labels.variant != "WT"]
    print("Starting non-wt length:", len(variants_without_WT))

    if len(variants_without_WT) == 0:
        raise ValueError("No non-WT variants found in labels.")

    if first_round_strategy == "random":
        random_mutants = _safe_sample_without_replacement(
            variants_without_WT,
            num_mutants_per_round,
            random_seed=random_seed,
        )
        iteration_zero_ids = random_mutants

    elif first_round_strategy == "diverse_medoids":
        from sklearn_extra.cluster import KMedoids

        if "WT" in embeddings.index:
            embeddings_without_WT = embeddings.drop("WT")
        else:
            embeddings_without_WT = embeddings.copy()

        print("embeddings:", len(embeddings))
        print("embeddings without wildtype:", len(embeddings_without_WT))

        num_clusters = min(num_mutants_per_round, len(embeddings_without_WT))
        if num_clusters == 0:
            iteration_zero_ids = []
        else:
            if embedding_type != "embeddings_pca":
                print("Performing PCA on embeddings")
                n_comp = min(10, embeddings_without_WT.shape[0], embeddings_without_WT.shape[1])
                if n_comp < 1:
                    pca_embeddings_reduced = embeddings_without_WT.to_numpy()
                else:
                    pca = PCA(n_components=n_comp)
                    pca_embeddings_reduced = pca.fit_transform(embeddings_without_WT)
            else:
                pca_embeddings_reduced = embeddings_without_WT.to_numpy()

            clusters = KMedoids(
                n_clusters=num_clusters,
                metric="euclidean",
                random_state=random_seed,
            ).fit(pca_embeddings_reduced)
            cluster_medoids = clusters.medoid_indices_
            selected_mutants = embeddings_without_WT.index[cluster_medoids].tolist()
            iteration_zero_ids = selected_mutants

    elif first_round_strategy == "explicit_variants":
        iteration_zero_ids = explicit_variants if explicit_variants is not None else []

    else:
        raise ValueError("Invalid first round search strategy.")

    iteration_zero = pd.DataFrame({"variant": iteration_zero_ids, "iteration": 0})
    if "WT" in labels["variant"].values:
        wt_df = pd.DataFrame({"variant": ["WT"], "iteration": [0]})
        iteration_zero = pd.concat([iteration_zero, wt_df], ignore_index=True)

    iteration_zero = iteration_zero.drop_duplicates(subset=["variant"], keep="first")
    this_round_variants = iteration_zero["variant"]

    labels_zero = pd.merge(labels, iteration_zero, on="variant", how="left")
    return labels_zero, iteration_zero, this_round_variants


def top_layer(
    iter_train,
    iter_test,
    embeddings_pd,
    labels_pd,
    measured_var,
    regression_type="randomforest",
    top_n=None,
    final_round=10,
    experimental=False,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    ucb_beta: float = DEFAULT_UCB_BETA,
    random_seed: int = 1,
    enable_tail_weighting: bool = DEFAULT_ENABLE_TAIL_WEIGHTING,
    tail_weight_mode: str = DEFAULT_TAIL_WEIGHT_MODE,
    tail_weight_quantile: float = DEFAULT_TAIL_WEIGHT_QUANTILE,
    tail_weight_strength: float = DEFAULT_TAIL_WEIGHT_STRENGTH,
    tail_weight_max: float = DEFAULT_TAIL_WEIGHT_MAX,
    enable_head_rerank: bool = DEFAULT_ENABLE_HEAD_RERANK,
    head_rerank_pool_size: int = DEFAULT_HEAD_RERANK_POOL_SIZE,
    head_rerank_weight: float = DEFAULT_HEAD_RERANK_WEIGHT,
    head_rerank_tail_quantile: float = DEFAULT_HEAD_RERANK_TAIL_QUANTILE,
    enable_success_head: bool = DEFAULT_ENABLE_SUCCESS_HEAD,
    success_label_mode: str = DEFAULT_SUCCESS_LABEL_MODE,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    success_positive_quantile: float = DEFAULT_SUCCESS_POSITIVE_QUANTILE,
):
    if experimental:
        label_variants = labels_pd["variant"].tolist()
        embedding_variants = embeddings_pd.index.tolist()
        if label_variants == embedding_variants:
            print("Embeddings and labels are aligned")
        else:
            print("Embeddings and labels are not aligned")
            print("Exiting.")
            return None

    embeddings_pd = embeddings_pd.copy()
    labels_pd = labels_pd.copy()

    embeddings_pd = _make_full_feature_matrix_unique(embeddings_pd)

    embeddings_pd = embeddings_pd.reset_index(drop=True)
    labels_pd = labels_pd.reset_index(drop=True)

    iteration = labels_pd["iteration"]
    labels = labels_pd

    idx_train = iteration[iteration.isin(iter_train)].index.to_numpy()
    if iter_test is not None:
        idx_test = iteration[iteration == iter_test].index.to_numpy()
    else:
        idx_test = iteration[iteration.isna()].index.to_numpy()

    X_all = _safe_fill_numeric(embeddings_pd)
    X_train_full = X_all.loc[idx_train, :].copy().reset_index(drop=True)
    X_test_full = X_all.loc[idx_test, :].copy().reset_index(drop=True)

    y_train = labels.loc[idx_train, measured_var].to_numpy()
    y_train_activity_scaled = labels.loc[idx_train, "activity_scaled"].to_numpy()
    y_train_activity_binary = labels.loc[idx_train, "activity_binary"].to_numpy()

    y_test = labels.loc[idx_test, measured_var].to_numpy()
    y_test_activity_scaled = labels.loc[idx_test, "activity_scaled"].to_numpy()
    y_test_activity_binary = labels.loc[idx_test, "activity_binary"].to_numpy()

    plm_cols, struct_cols = _split_feature_blocks(X_all)

    has_plm = len(plm_cols) > 0
    has_struct = len(struct_cols) > 0

    if has_plm:
        X_train_plm = _rename_block_columns(X_train_full[plm_cols], "plm")
        X_test_plm = _rename_block_columns(X_test_full[plm_cols], "plm")
    else:
        X_train_plm = pd.DataFrame(index=X_train_full.index)
        X_test_plm = pd.DataFrame(index=X_test_full.index)

    if has_struct:
        X_train_struct = _rename_block_columns(X_train_full[struct_cols], "struct")
        X_test_struct = _rename_block_columns(X_test_full[struct_cols], "struct")
    else:
        X_train_struct = pd.DataFrame(index=X_train_full.index)
        X_test_struct = pd.DataFrame(index=X_test_full.index)

    train_sample_weights = _compute_tail_sample_weights(
        y_train=y_train,
        y_train_activity_scaled=y_train_activity_scaled,
        enable_tail_weighting=enable_tail_weighting,
        tail_weight_mode=tail_weight_mode,
        tail_weight_quantile=tail_weight_quantile,
        tail_weight_strength=tail_weight_strength,
        tail_weight_max=tail_weight_max,
    )

    use_hybrid = (
        regression_type == "hybrid"
        and has_plm
        and has_struct
        and X_train_plm.shape[1] > 0
        and X_train_struct.shape[1] > 0
    )

    if use_hybrid:
        print(
            f"Using bootstrap hybrid residual fusion: "
            f"PLM({X_train_plm.shape[1]}) + Struct({X_train_struct.shape[1]}) "
            f"| ensemble_size={ensemble_size}"
        )
        (
            model,
            y_pred_train_base,
            y_pred_test_base,
            y_std_train,
            y_std_test,
            alpha,
        ) = _fit_predict_bootstrap_ensemble(
            regression_type="hybrid",
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_train=y_train,
            use_hybrid=True,
            X_train_plm=X_train_plm,
            X_test_plm=X_test_plm,
            X_train_struct=X_train_struct,
            X_test_struct=X_test_struct,
            ensemble_size=ensemble_size,
            bootstrap_sample_frac=bootstrap_sample_frac,
            random_seed=random_seed,
            train_sample_weights=train_sample_weights,
        )
    else:
        if regression_type == "hybrid":
            fallback_regression = "ridge" if has_plm else "randomforest"
        else:
            fallback_regression = regression_type

        if X_train_full.shape[1] == 0:
            raise ValueError("No features available for training.")

        print(
            f"Using bootstrap single-model top layer: "
            f"{fallback_regression} | ensemble_size={ensemble_size}"
        )

        (
            model,
            y_pred_train_base,
            y_pred_test_base,
            y_std_train,
            y_std_test,
            alpha,
        ) = _fit_predict_bootstrap_ensemble(
            regression_type=fallback_regression,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_train=y_train,
            use_hybrid=False,
            ensemble_size=ensemble_size,
            bootstrap_sample_frac=bootstrap_sample_frac,
            random_seed=random_seed,
            train_sample_weights=train_sample_weights,
        )

    if has_plm and X_train_plm.shape[1] > 0 and X_test_plm.shape[1] > 0:
        X_train_rerank = X_train_plm
        X_test_rerank = X_test_plm
    else:
        X_train_rerank = X_train_full
        X_test_rerank = X_test_full

    y_train_ref_for_tail = _make_tail_reference(
        y_train=y_train,
        y_train_activity_scaled=y_train_activity_scaled,
    )

    (
        y_pred_test_final,
        head_rerank_score_test,
        head_rerank_correction_test,
        head_rerank_applied_test,
    ) = _apply_head_rerank_to_test_predictions(
        base_pred_test=y_pred_test_base,
        X_train_rerank=X_train_rerank,
        y_train_ref=y_train_ref_for_tail,
        X_test_rerank=X_test_rerank,
        enable_head_rerank=enable_head_rerank,
        head_rerank_pool_size=head_rerank_pool_size,
        head_rerank_weight=head_rerank_weight,
        head_rerank_tail_quantile=head_rerank_tail_quantile,
    )

    y_pred_train_final = np.asarray(y_pred_train_base, dtype=float)
    head_rerank_score_train = np.full(len(y_pred_train_final), np.nan)
    head_rerank_correction_train = np.zeros(len(y_pred_train_final), dtype=float)
    head_rerank_applied_train = np.zeros(len(y_pred_train_final), dtype=bool)

    if enable_success_head:
        y_success_train = _make_success_binary_labels(
            y_train=y_train,
            y_train_activity_scaled=y_train_activity_scaled,
            y_train_activity_binary=y_train_activity_binary,
            success_label_mode=success_label_mode,
            success_threshold=success_threshold,
            success_positive_quantile=success_positive_quantile,
        )

        print(
            f"Using success-aware blended head: "
            f"mode={success_label_mode}, threshold={success_threshold}, "
            f"positive_rate={float(np.mean(y_success_train)) if len(y_success_train) else np.nan:.3f}, "
            f"ensemble_size={ensemble_size}"
        )

        (
            success_model,
            pred_success_prob_cls_train,
            pred_success_prob_cls_test,
            pred_success_prob_cls_std_train,
            pred_success_prob_cls_std_test,
        ) = _fit_predict_success_bootstrap_ensemble(
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            y_success_train=y_success_train,
            ensemble_size=ensemble_size,
            bootstrap_sample_frac=bootstrap_sample_frac,
            random_seed=random_seed,
        )

        (
            pred_success_prob_reg_train,
            pred_success_prob_reg_test,
            pred_success_prob_reg_std_train,
            pred_success_prob_reg_std_test,
            success_regression_threshold,
        ) = _compute_regression_success_probability(
            model_bundle=model,
            pred_mean_train=y_pred_train_base,
            pred_mean_test=y_pred_test_base,
            pred_std_train=y_std_train,
            pred_std_test=y_std_test,
            y_train=y_train,
            y_success_train=y_success_train,
            success_positive_quantile=success_positive_quantile,
        )

        (
            success_cls_blend_weight,
            success_train_n_pos,
            success_train_n_neg,
            success_train_pos_frac,
        ) = _resolve_success_blend_weight(
            y_success_train=y_success_train,
            success_model=success_model,
        )

        (
            pred_success_prob_train,
            pred_success_prob_test,
            pred_success_prob_std_train,
            pred_success_prob_std_test,
        ) = _blend_success_probabilities(
            prob_cls_train=pred_success_prob_cls_train,
            prob_cls_test=pred_success_prob_cls_test,
            std_cls_train=pred_success_prob_cls_std_train,
            std_cls_test=pred_success_prob_cls_std_test,
            prob_reg_train=pred_success_prob_reg_train,
            prob_reg_test=pred_success_prob_reg_test,
            std_reg_train=pred_success_prob_reg_std_train,
            std_reg_test=pred_success_prob_reg_std_test,
            cls_weight=success_cls_blend_weight,
        )

        print(
            f"Success blended head diagnostics: "
            f"n_pos={success_train_n_pos}, n_neg={success_train_n_neg}, "
            f"cls_weight={success_cls_blend_weight:.2f}, "
            f"reg_threshold={success_regression_threshold:.6g}"
        )

        try:
            if isinstance(model, dict):
                model["success_head"] = success_model
                model["success_head_blend"] = {
                    "cls_weight": float(success_cls_blend_weight),
                    "reg_weight": float(1.0 - success_cls_blend_weight),
                    "regression_threshold": float(success_regression_threshold),
                    "train_n_pos": int(success_train_n_pos),
                    "train_n_neg": int(success_train_n_neg),
                    "train_pos_frac": float(success_train_pos_frac),
                }
            else:
                model = {
                    "type": "regression_with_success_head",
                    "regression_model": model,
                    "success_head": success_model,
                    "success_head_blend": {
                        "cls_weight": float(success_cls_blend_weight),
                        "reg_weight": float(1.0 - success_cls_blend_weight),
                        "regression_threshold": float(success_regression_threshold),
                        "train_n_pos": int(success_train_n_pos),
                        "train_n_neg": int(success_train_n_neg),
                        "train_pos_frac": float(success_train_pos_frac),
                    },
                }
        except Exception:
            pass

        success_head_enabled_train = np.ones(len(y_pred_train_final), dtype=int)
        success_head_enabled_test = np.ones(len(y_pred_test_final), dtype=int)
    else:
        pred_success_prob_cls_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_cls_test = np.full(len(y_pred_test_final), np.nan, dtype=float)
        pred_success_prob_cls_std_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_cls_std_test = np.full(len(y_pred_test_final), np.nan, dtype=float)

        pred_success_prob_reg_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_reg_test = np.full(len(y_pred_test_final), np.nan, dtype=float)
        pred_success_prob_reg_std_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_reg_std_test = np.full(len(y_pred_test_final), np.nan, dtype=float)

        pred_success_prob_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_test = np.full(len(y_pred_test_final), np.nan, dtype=float)
        pred_success_prob_std_train = np.full(len(y_pred_train_final), np.nan, dtype=float)
        pred_success_prob_std_test = np.full(len(y_pred_test_final), np.nan, dtype=float)

        success_regression_threshold = np.nan
        success_cls_blend_weight = np.nan
        success_train_n_pos = 0
        success_train_n_neg = 0
        success_train_pos_frac = np.nan

        success_head_enabled_train = np.zeros(len(y_pred_train_final), dtype=int)
        success_head_enabled_test = np.zeros(len(y_pred_test_final), dtype=int)

    success_score_train = np.clip(
        pred_success_prob_train - float(DEFAULT_SUCCESS_SCORE_RISK_PENALTY) * pred_success_prob_std_train,
        0.0,
        1.0,
    )
    success_score_test = np.clip(
        pred_success_prob_test - float(DEFAULT_SUCCESS_SCORE_RISK_PENALTY) * pred_success_prob_std_test,
        0.0,
        1.0,
    )

    train_error = _safe_mse(y_train, y_pred_train_final)
    test_error = None if experimental else _safe_mse(y_test, y_pred_test_final)
    train_r_squared = _safe_r2(y_train, y_pred_train_final)
    test_r_squared = None if experimental else _safe_r2(y_test, y_pred_test_final)

    dist_metric_train = _safe_min_pairwise_distance(X_train_full, X_test_full)
    dist_metric_test = None if experimental else _safe_min_pairwise_distance(X_test_full, X_train_full)

    novelty_train = _safe_min_pairwise_distance_excluding_self(X_train_full)

    novelty_test = None if experimental else _safe_min_pairwise_distance(X_test_full, X_train_full)

    ucb_score_train = y_pred_train_final + float(ucb_beta) * y_std_train
    ucb_score_test = None if experimental else (y_pred_test_final + float(ucb_beta) * y_std_test)

    df_train = pd.DataFrame(
        {
            "variant": labels.loc[idx_train, "variant"].values,
            "y_pred": y_pred_train_final,
            "pred_mean": y_pred_train_final,
            "pred_mean_base": y_pred_train_base,
            "y_actual": y_train,
            "y_actual_scaled": y_train_activity_scaled,
            "y_actual_binary": y_train_activity_binary,
            "dist_metric": dist_metric_train,
            "novelty_to_tested": novelty_train,
            "std_predictions": y_std_train,
            "pred_std": y_std_train,
            "ucb_score_base": ucb_score_train,
            "pred_success_prob": pred_success_prob_train,
            "pred_success_prob_cls": pred_success_prob_cls_train,
            "pred_success_prob_reg": pred_success_prob_reg_train,
            "pred_success_prob_std": pred_success_prob_std_train,
            "pred_success_prob_cls_std": pred_success_prob_cls_std_train,
            "pred_success_prob_reg_std": pred_success_prob_reg_std_train,
            "success_score_base": success_score_train,
            "success_head_enabled": success_head_enabled_train,
            "success_regression_threshold": success_regression_threshold,
            "success_cls_blend_weight": success_cls_blend_weight,
            "success_train_n_pos": success_train_n_pos,
            "success_train_n_neg": success_train_n_neg,
            "success_train_pos_frac": success_train_pos_frac,
            "tail_sample_weight": train_sample_weights,
            "head_rerank_score": head_rerank_score_train,
            "head_rerank_correction": head_rerank_correction_train,
            "head_rerank_applied": head_rerank_applied_train.astype(int),
        }
    )

    df_test = pd.DataFrame(
        {
            "variant": labels.loc[idx_test, "variant"].values,
            "y_pred": y_pred_test_final,
            "pred_mean": y_pred_test_final,
            "pred_mean_base": y_pred_test_base,
            "y_actual": y_test,
            "y_actual_scaled": y_test_activity_scaled,
            "y_actual_binary": y_test_activity_binary,
            "dist_metric": dist_metric_test,
            "novelty_to_tested": novelty_test,
            "std_predictions": y_std_test,
            "pred_std": y_std_test,
            "ucb_score_base": ucb_score_test,
            "pred_success_prob": pred_success_prob_test,
            "pred_success_prob_cls": pred_success_prob_cls_test,
            "pred_success_prob_reg": pred_success_prob_reg_test,
            "pred_success_prob_std": pred_success_prob_std_test,
            "pred_success_prob_cls_std": pred_success_prob_cls_std_test,
            "pred_success_prob_reg_std": pred_success_prob_reg_std_test,
            "success_score_base": success_score_test,
            "success_head_enabled": success_head_enabled_test,
            "success_regression_threshold": success_regression_threshold,
            "success_cls_blend_weight": success_cls_blend_weight,
            "success_train_n_pos": success_train_n_pos,
            "success_train_n_neg": success_train_n_neg,
            "success_train_pos_frac": success_train_pos_frac,
            "tail_sample_weight": np.nan,
            "head_rerank_score": head_rerank_score_test,
            "head_rerank_correction": head_rerank_correction_test,
            "head_rerank_applied": head_rerank_applied_test.astype(int),
        }
    )

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_sorted_all = df_all.sort_values("y_pred", ascending=False).reset_index(drop=True)

    this_round_variants = df_train["variant"]


    if top_n is not None:
        top_k = min(int(top_n), len(df_sorted_all))
    else:
        top_k = min(int(final_round), len(df_sorted_all))

    top_block = df_sorted_all.head(top_k).copy()

    median_activity_scaled = top_block["y_actual_scaled"].median() if len(top_block) > 0 else np.nan
    top_activity_scaled = top_block["y_actual_scaled"].max() if len(top_block) > 0 else np.nan

    if len(top_block) > 0 and pd.notna(top_activity_scaled):
        top_variant_candidates = top_block.loc[
            top_block["y_actual_scaled"] == top_activity_scaled, "variant"
        ].tolist()
        top_variant = top_variant_candidates[0] if len(top_variant_candidates) > 0 else top_block.iloc[0]["variant"]
    else:
        top_variant = top_block.iloc[0]["variant"] if len(top_block) > 0 else None

    top_final_round_variants = ",".join(top_block["variant"].tolist()) if len(top_block) > 0 else ""

    spearman_corr = _safe_spearman(df_sorted_all["y_actual"].values, df_sorted_all["y_pred"].values)
    activity_binary_percentage = top_block["y_actual_binary"].mean() if len(top_block) > 0 else np.nan

    if experimental:
        return this_round_variants, df_test, df_sorted_all
    else:
        return (
            train_error,
            test_error,
            train_r_squared,
            test_r_squared,
            alpha,
            median_activity_scaled,
            top_activity_scaled,
            top_variant,
            top_final_round_variants,
            activity_binary_percentage,
            spearman_corr,
            df_test,
            this_round_variants,
        )
