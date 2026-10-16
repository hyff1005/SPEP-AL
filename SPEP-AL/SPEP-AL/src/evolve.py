import os
import random
import time
import pandas as pd
import numpy as np
import re
from typing import Any, List, Optional, Tuple, Dict, Union

from scipy.spatial.distance import cdist

from evolvepro.src.data import (
    load_dms_data,
    load_experimental_embeddings,
    load_experimental_data,
    create_iteration_dataframes,
)
from evolvepro.src.utils import pca_embeddings
from evolvepro.src.model import first_round, top_layer


DEFAULT_ENSEMBLE_SIZE = 16
DEFAULT_BOOTSTRAP_SAMPLE_FRAC = 1.0
DEFAULT_UCB_BETA = 0.5
DEFAULT_NOVELTY_WEIGHT = 0.2
DEFAULT_REDUNDANCY_WEIGHT = 0.2
DEFAULT_THOMPSON_SCALE = 1.0

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
DEFAULT_SUCCESS_THRESHOLD = 0.5
DEFAULT_SUCCESS_POSITIVE_QUANTILE = 0.70
DEFAULT_SUCCESS_WEIGHT = 0.0
DEFAULT_SUCCESS_FILTER_QUANTILE = 0.0
DEFAULT_SUCCESS_MODE = "score_bonus"

DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT = 0.75
DEFAULT_SUCCESS_ACQ_PRED_WEIGHT = 0.20
DEFAULT_SUCCESS_ACQ_STD_PENALTY = 0.10
DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT = 0.03
DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE = 0.50
DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE = 0.50

DEFAULT_SAFE_MAX_REPLACE = 1
DEFAULT_SAFE_CANDIDATE_POOL_QUANTILE = 0.80
DEFAULT_SAFE_STD_PENALTY = 0.10
DEFAULT_SAFE_NOVELTY_WEIGHT = 0.01
DEFAULT_SAFE_BATCH_MARGIN = 0.00
DEFAULT_SAFE_REQUIRE_BATCH_SCORE = True

DEFAULT_MIXED_EXPLOIT_FRAC = 0.75

DEFAULT_MIXED_SCHEDULE_A = "1-3:0.75,4-6:0.8125,7-10:0.875"
DEFAULT_MIXED_SCHEDULE_B = "1-4:0.8125,5-10:0.875"

DEFAULT_MIXED_SCHEDULE_PRESETS = {
    "schedule_a": DEFAULT_MIXED_SCHEDULE_A,
    "a": DEFAULT_MIXED_SCHEDULE_A,
    "three_stage": DEFAULT_MIXED_SCHEDULE_A,
    "schedule_b": DEFAULT_MIXED_SCHEDULE_B,
    "b": DEFAULT_MIXED_SCHEDULE_B,
    "two_stage": DEFAULT_MIXED_SCHEDULE_B,
}

DEFAULT_PARETO_POOL_MULTIPLIER = 4.0
DEFAULT_PARETO_LAMBDA_VALUES = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00)
DEFAULT_PARETO_BALANCE_RULE = "product"
DEFAULT_PARETO_MIN_QUALITY_RETENTION = 0.95
DEFAULT_PARETO_LOCAL_SEARCH = True
DEFAULT_PARETO_LOCAL_SEARCH_STEPS = 50
DEFAULT_PARETO_ENTROPY_BASE = "batch"
DEFAULT_PARETO_FORCE_TOP1 = True


def _parse_pca_components_from_name(embedding_type: str, default: int = 10) -> int:
    match = re.search(r"pca_(\d+)", embedding_type)
    if match:
        return int(match.group(1))
    return default


def _add_prefix_to_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None:
        return None

    df = df.copy()
    new_cols = []

    for col in df.columns:
        col_str = str(col)

        if col_str.startswith(f"{prefix}__"):
            new_cols.append(col_str)
        else:
            new_cols.append(f"{prefix}__{col_str}")

    df.columns = new_cols
    return df


def _prepare_plm_block(plm_embeddings: pd.DataFrame) -> pd.DataFrame:
    return _add_prefix_to_columns(plm_embeddings, prefix="plm")


def _prepare_struct_block(struct_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if struct_df is None:
        return None
    return _add_prefix_to_columns(struct_df, prefix="struct")


def _build_embedding_pool(
    plm_embeddings: pd.DataFrame,
    struct_df: Optional[pd.DataFrame],
    embedding_types: List[str],
) -> Dict[str, pd.DataFrame]:
    embeddings_list: Dict[str, pd.DataFrame] = {}

    plm_block = _prepare_plm_block(plm_embeddings)
    struct_block = _prepare_struct_block(struct_df)

    embeddings_list["embeddings"] = plm_block

    if struct_block is not None:
        embeddings_list["embeddings_struct_only"] = struct_block
        embeddings_list["embeddings_struct_fusion"] = pd.concat(
            [plm_block, struct_block], axis=1
        )

    for etype in embedding_types:
        if etype in embeddings_list:
            continue

        if etype == "struct_only":
            if struct_block is not None:
                embeddings_list[etype] = struct_block
            continue

        if "pca" not in etype:
            continue

        n_comp = _parse_pca_components_from_name(etype, default=10)
        print(f"Generating PCA embedding for '{etype}' with {n_comp} components...")

        plm_pca = pca_embeddings(plm_embeddings, n_components=n_comp)
        plm_pca_block = _prepare_plm_block(plm_pca)

        if "struct_fusion" in etype:
            if struct_block is None:
                print(f"Warning: requested '{etype}' but structural features are unavailable. Skipping.")
                continue

            fused_pca = pd.concat([plm_pca_block, struct_block], axis=1)
            embeddings_list[etype] = fused_pca
            print(
                f"Created '{etype}': PCA({plm_pca_block.shape[1]}) + Struct({struct_block.shape[1]}) "
                f"=> Total Dim {fused_pca.shape[1]}"
            )
        else:
            embeddings_list[etype] = plm_pca_block
            print(f"Created '{etype}': PCA({plm_pca_block.shape[1]})")

    return embeddings_list

def _safe_pick_random_variants(variants: List[str], k: int, seed: int) -> List[str]:
    variants = list(variants)
    if len(variants) == 0:
        return []
    k = min(k, len(variants))
    rng = random.Random(seed)
    return rng.sample(variants, k)


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _resolve_pred_col(df: pd.DataFrame) -> str:
    if "pred_mean" in df.columns:
        return "pred_mean"
    if "y_pred" in df.columns:
        return "y_pred"
    raise ValueError("No prediction mean column found. Expected 'pred_mean' or 'y_pred'.")


def _resolve_base_pred_col(df: pd.DataFrame) -> Optional[str]:
    if "pred_mean_base" in df.columns:
        return "pred_mean_base"
    if "pred_mean" in df.columns:
        return "pred_mean"
    if "y_pred" in df.columns:
        return "y_pred"
    return None


def _resolve_std_col(df: pd.DataFrame) -> Optional[str]:
    if "pred_std" in df.columns:
        return "pred_std"
    if "std_predictions" in df.columns:
        return "std_predictions"
    return None


def _resolve_novelty_col(df: pd.DataFrame) -> Optional[str]:
    if "novelty_to_tested" in df.columns:
        return "novelty_to_tested"
    if "dist_metric" in df.columns:
        return "dist_metric"
    return None


def _resolve_success_prob_col(df: pd.DataFrame) -> Optional[str]:
    if df is None:
        return None
    if "pred_success_prob" in df.columns:
        return "pred_success_prob"
    if "success_score_base" in df.columns:
        return "success_score_base"
    return None


def _resolve_success_prob_std_col(df: pd.DataFrame) -> Optional[str]:
    if df is None:
        return None
    if "pred_success_prob_std" in df.columns:
        return "pred_success_prob_std"
    return None


def _success_head_is_available(df: pd.DataFrame) -> bool:
    col = _resolve_success_prob_col(df)
    if col is None or col not in df.columns:
        return False
    vals = _safe_numeric_series(df[col])
    return bool(vals.notna().any())


def _normalize_success_mode(success_mode: Optional[str]) -> str:
    mode = str(success_mode or "off").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "bonus": "score_bonus",
        "score": "score_bonus",
        "score_bonus": "score_bonus",
        "filter": "filter_then_score",
        "filter_then_score": "filter_then_score",
        "filter_score": "filter_then_score",
    }
    return aliases.get(mode, mode)


def _apply_success_filter_to_candidates(
    df_test_new: pd.DataFrame,
    k: int,
    success_filter_quantile: float = DEFAULT_SUCCESS_FILTER_QUANTILE,
    success_mode: str = DEFAULT_SUCCESS_MODE,
) -> pd.DataFrame:
    if df_test_new is None or df_test_new.empty:
        return df_test_new

    mode = _normalize_success_mode(success_mode)
    if mode != "filter_then_score":
        return df_test_new

    try:
        q = float(success_filter_quantile)
    except Exception:
        q = 0.0
    q = float(min(0.95, max(0.0, q)))
    if q <= 0.0:
        return df_test_new

    success_col = _resolve_success_prob_col(df_test_new)
    if success_col is None:
        return df_test_new

    work = df_test_new.copy()
    success_vals = _safe_numeric_series(work[success_col])
    valid = success_vals.notna()
    if valid.sum() < max(2, int(k)):
        return df_test_new

    cutoff = float(success_vals[valid].quantile(q))
    filtered = work[success_vals >= cutoff].copy()

    if len(filtered) < min(int(k), len(work)):
        return df_test_new

    return filtered.reset_index(drop=True)


def _rank_normalize(values: pd.Series) -> pd.Series:
    v = _safe_numeric_series(values).copy()
    if len(v) == 0:
        return pd.Series(dtype=float, index=values.index if hasattr(values, "index") else None)

    valid = v.notna()
    out = pd.Series(np.zeros(len(v), dtype=float), index=v.index)

    if valid.sum() == 0:
        return out

    ranked = v[valid].rank(method="average", pct=True)
    out.loc[valid] = ranked.astype(float)
    out.loc[~valid] = 0.0
    return out


def _subset_embeddings_by_variants(
    embeddings: pd.DataFrame,
    variants: List[str],
) -> Optional[pd.DataFrame]:
    if embeddings is None or len(variants) == 0:
        return None

    if isinstance(embeddings.index, pd.Index):
        common = [v for v in variants if v in embeddings.index]
        if len(common) == 0:
            return None
        X = embeddings.loc[common].copy()
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians = X.median(axis=0, skipna=True).fillna(0.0)
        X = X.fillna(medians)
        return X

    return None


def _pairwise_distance_matrix(X: pd.DataFrame) -> Optional[np.ndarray]:
    if X is None or len(X) == 0:
        return None
    return cdist(X.values, X.values, metric="euclidean")


def _avg_pairwise_distance(X: pd.DataFrame) -> float:
    if X is None or len(X) < 2:
        return np.nan
    dmat = cdist(X.values, X.values, metric="euclidean")
    iu = np.triu_indices_from(dmat, k=1)
    vals = dmat[iu]
    return float(np.mean(vals)) if len(vals) > 0 else np.nan


def _avg_min_pairwise_distance(X: pd.DataFrame) -> float:
    if X is None or len(X) < 2:
        return np.nan
    dmat = cdist(X.values, X.values, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    mins = dmat.min(axis=1)
    return float(np.mean(mins)) if len(mins) > 0 else np.nan


def _ordered_unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _split_mixed_counts(
    k: int,
    mixed_exploit_frac: float = DEFAULT_MIXED_EXPLOIT_FRAC,
) -> Tuple[int, int]:
    if k <= 0:
        return 0, 0
    if k == 1:
        return 1, 0

    exploit_k = int(np.floor(k * float(mixed_exploit_frac)))
    exploit_k = max(1, exploit_k)
    exploit_k = min(exploit_k, k - 1)

    explore_k = k - exploit_k
    return exploit_k, explore_k


def _clamp_mixed_exploit_frac(mixed_exploit_frac: Union[int, float]) -> float:
    try:
        value = float(mixed_exploit_frac)
    except Exception:
        return float(DEFAULT_MIXED_EXPLOIT_FRAC)

    if np.isnan(value):
        return float(DEFAULT_MIXED_EXPLOIT_FRAC)

    return float(min(1.0, max(0.0, value)))


def _normalize_schedule_spec_string(
    schedule_spec: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]]
) -> Optional[str]:
    if schedule_spec is None:
        return None

    if isinstance(schedule_spec, str):
        schedule_str = schedule_spec.strip()
        if schedule_str == "":
            return None

        key = schedule_str.lower()
        if key.startswith("preset:"):
            key = key.split(":", 1)[1].strip().lower()
        if key in DEFAULT_MIXED_SCHEDULE_PRESETS:
            return DEFAULT_MIXED_SCHEDULE_PRESETS[key]
        return schedule_str

    if isinstance(schedule_spec, dict):
        parts = []
        for round_num in sorted(schedule_spec):
            frac = _clamp_mixed_exploit_frac(schedule_spec[round_num])
            parts.append(f"{int(round_num)}:{frac:g}")
        return ",".join(parts)

    if isinstance(schedule_spec, (list, tuple)):
        parts = []
        for idx, frac in enumerate(schedule_spec, start=1):
            parts.append(f"{idx}:{_clamp_mixed_exploit_frac(frac):g}")
        return ",".join(parts)

    return str(schedule_spec)


def _parse_round_token(round_token: str) -> List[int]:
    round_token = str(round_token).strip()
    if round_token == "":
        return []

    if "-" in round_token:
        left, right = round_token.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))

    return [int(round_token)]


def _parse_mixed_exploit_schedule(
    schedule_spec: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]],
    total_rounds: Optional[int] = None,
) -> Dict[int, float]:
    schedule_map: Dict[int, float] = {}

    if schedule_spec is None:
        return schedule_map

    if isinstance(schedule_spec, dict):
        for round_num, frac in schedule_spec.items():
            r = int(round_num)
            if r < 1:
                continue
            if total_rounds is not None and r > int(total_rounds):
                continue
            schedule_map[r] = _clamp_mixed_exploit_frac(frac)
        return schedule_map

    if isinstance(schedule_spec, (list, tuple)):
        for idx, frac in enumerate(schedule_spec, start=1):
            if total_rounds is not None and idx > int(total_rounds):
                break
            schedule_map[idx] = _clamp_mixed_exploit_frac(frac)
        return schedule_map

    if not isinstance(schedule_spec, str):
        return schedule_map

    schedule_str = schedule_spec.strip()
    if schedule_str == "":
        return schedule_map

    key = schedule_str.lower()
    if key.startswith("preset:"):
        key = key.split(":", 1)[1].strip().lower()
    if key in DEFAULT_MIXED_SCHEDULE_PRESETS:
        schedule_str = DEFAULT_MIXED_SCHEDULE_PRESETS[key]

    tokens = re.split(r"[;,]+", schedule_str)
    for token in tokens:
        token = token.strip()
        if token == "":
            continue

        if ":" not in token:
            raise ValueError(
                f"Invalid mixed_exploit_schedule token: '{token}'. "
                f"Expected format like '1-3:0.75'."
            )

        round_token, frac_token = token.split(":", 1)
        frac = _clamp_mixed_exploit_frac(frac_token)

        for round_num in _parse_round_token(round_token):
            if round_num < 1:
                continue
            if total_rounds is not None and round_num > int(total_rounds):
                continue
            schedule_map[int(round_num)] = frac

    return schedule_map


def _resolve_mixed_exploit_frac_for_round(
    round_num: int,
    mixed_exploit_frac: float = DEFAULT_MIXED_EXPLOIT_FRAC,
    mixed_exploit_schedule: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]] = None,
    parsed_schedule_map: Optional[Dict[int, float]] = None,
) -> float:
    default_frac = _clamp_mixed_exploit_frac(mixed_exploit_frac)

    if round_num is None or int(round_num) < 1:
        return default_frac

    if parsed_schedule_map is not None:
        return float(parsed_schedule_map.get(int(round_num), default_frac))

    if mixed_exploit_schedule is None:
        return default_frac

    schedule_map = _parse_mixed_exploit_schedule(mixed_exploit_schedule)
    return float(schedule_map.get(int(round_num), default_frac))

def _build_candidate_score_table(
    df_test_new: pd.DataFrame,
    ucb_beta: float,
    novelty_weight: float,
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
) -> pd.DataFrame:
    pred_col = _resolve_pred_col(df_test_new)
    std_col = _resolve_std_col(df_test_new)
    novelty_col = _resolve_novelty_col(df_test_new)
    success_col = _resolve_success_prob_col(df_test_new)

    work = df_test_new.copy().reset_index(drop=True)

    work["pred_mean_val"] = _safe_numeric_series(work[pred_col]).fillna(-np.inf)

    if std_col is not None:
        work["pred_std_val"] = _safe_numeric_series(work[std_col]).fillna(0.0)
    else:
        work["pred_std_val"] = 0.0

    if novelty_col is not None:
        work["novelty_val"] = _safe_numeric_series(work[novelty_col]).fillna(0.0)
    else:
        work["novelty_val"] = 0.0

    if success_col is not None:
        work["success_prob_val"] = _safe_numeric_series(work[success_col]).fillna(0.0)
    else:
        work["success_prob_val"] = 0.0

    work["ucb_raw"] = work["pred_mean_val"] + float(ucb_beta) * work["pred_std_val"]

    work["pred_rank"] = _rank_normalize(work["pred_mean_val"])
    work["std_rank"] = _rank_normalize(work["pred_std_val"])
    work["novelty_rank"] = _rank_normalize(work["novelty_val"])
    work["success_rank"] = _rank_normalize(work["success_prob_val"])

    work["acq_base_score"] = (
        work["pred_rank"]
        + float(ucb_beta) * work["std_rank"]
        + float(novelty_weight) * work["novelty_rank"]
        + float(success_weight) * work["success_rank"]
    )

    return work


def _build_success_first_score_table(
    df_test_new: pd.DataFrame,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT,
) -> pd.DataFrame:
    if df_test_new is None or df_test_new.empty:
        return pd.DataFrame()

    pred_col = _resolve_pred_col(df_test_new)
    std_col = _resolve_std_col(df_test_new)
    novelty_col = _resolve_novelty_col(df_test_new)
    success_col = _resolve_success_prob_col(df_test_new)

    work = df_test_new.copy().reset_index(drop=True)
    work["pred_mean_val"] = _safe_numeric_series(work[pred_col]).fillna(-np.inf)

    if std_col is not None:
        work["pred_std_val"] = _safe_numeric_series(work[std_col]).fillna(0.0)
    else:
        work["pred_std_val"] = 0.0

    if novelty_col is not None:
        work["novelty_val"] = _safe_numeric_series(work[novelty_col]).fillna(0.0)
    else:
        work["novelty_val"] = 0.0

    work["pred_rank"] = _rank_normalize(work["pred_mean_val"])
    work["std_rank"] = _rank_normalize(work["pred_std_val"])
    work["novelty_rank"] = _rank_normalize(work["novelty_val"])

    if success_col is not None:
        success_vals = _safe_numeric_series(work[success_col])
        if success_vals.notna().any():
            work["success_prob_val"] = success_vals.fillna(success_vals.median())
            work["success_head_available"] = 1.0
        else:
            work["success_prob_val"] = work["pred_rank"]
            work["success_head_available"] = 0.0
    else:
        work["success_prob_val"] = work["pred_rank"]
        work["success_head_available"] = 0.0

    work["success_rank"] = _rank_normalize(work["success_prob_val"])

    sw = float(success_acq_success_weight)
    pw = float(success_acq_pred_weight)
    sp = float(success_acq_std_penalty)
    nw = float(success_acq_novelty_weight)

    work["success_first_score"] = (
        sw * work["success_rank"]
        + pw * work["pred_rank"]
        - sp * work["std_rank"]
        + nw * work["novelty_rank"]
    )

    work["success_first_score_base"] = work["success_first_score"]

    return work


def _select_success_topn(
    df_test_new: pd.DataFrame,
    k: int,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = 0.0,
) -> List[str]:
    if df_test_new is None or df_test_new.empty:
        return []

    work = _build_success_first_score_table(
        df_test_new=df_test_new,
        success_acq_success_weight=success_acq_success_weight,
        success_acq_pred_weight=success_acq_pred_weight,
        success_acq_std_penalty=success_acq_std_penalty,
        success_acq_novelty_weight=success_acq_novelty_weight,
    )
    return work.sort_values(by="success_first_score", ascending=False).head(k)["variant"].tolist()


def _apply_success_explore_gate(
    work: pd.DataFrame,
    k: int,
    success_explore_success_quantile: float = DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE,
    success_explore_pred_quantile: float = DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE,
) -> Tuple[pd.DataFrame, bool, float, float]:
    if work is None or work.empty:
        return work, False, np.nan, np.nan

    q_success = float(min(0.95, max(0.0, success_explore_success_quantile)))
    q_pred = float(min(0.95, max(0.0, success_explore_pred_quantile)))

    success_vals = _safe_numeric_series(work["success_prob_val"])
    pred_vals = _safe_numeric_series(work["pred_mean_val"])

    success_cutoff = float(success_vals.quantile(q_success)) if success_vals.notna().any() else -np.inf
    pred_cutoff = float(pred_vals.quantile(q_pred)) if pred_vals.notna().any() else -np.inf

    gated = work[(success_vals >= success_cutoff) & (pred_vals >= pred_cutoff)].copy()

    if len(gated) < min(int(k), len(work)):
        return work.copy(), False, success_cutoff, pred_cutoff

    return gated.reset_index(drop=True), bool(len(gated) < len(work)), success_cutoff, pred_cutoff


def _select_success_constrained_mixed(
    df_test_new: pd.DataFrame,
    embeddings: pd.DataFrame,
    k: int,
    mixed_exploit_frac: float,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT,
    success_explore_success_quantile: float = DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE,
    success_explore_pred_quantile: float = DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE,
    redundancy_weight: float = DEFAULT_REDUNDANCY_WEIGHT,
) -> Tuple[List[str], Dict[str, Any]]:
    if df_test_new is None or df_test_new.empty:
        return [], {
            "selected_exploit_count": 0,
            "selected_explore_count": 0,
            "success_gate_applied": False,
            "success_gate_success_cutoff": np.nan,
            "success_gate_pred_cutoff": np.nan,
            "success_head_available_for_selection": False,
        }

    n_available = len(df_test_new)
    k = min(k, n_available)

    exploit_k, explore_k = _split_mixed_counts(
        k=k,
        mixed_exploit_frac=mixed_exploit_frac,
    )

    exploit_variants = _select_success_topn(
        df_test_new=df_test_new,
        k=exploit_k,
        success_acq_success_weight=success_acq_success_weight,
        success_acq_pred_weight=success_acq_pred_weight,
        success_acq_std_penalty=success_acq_std_penalty,
        success_acq_novelty_weight=0.0,
    )

    remaining_df = df_test_new[~df_test_new["variant"].isin(exploit_variants)].copy()

    if explore_k <= 0 or remaining_df.empty:
        selected = exploit_variants[:k]
        return selected, {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
            "success_gate_applied": False,
            "success_gate_success_cutoff": np.nan,
            "success_gate_pred_cutoff": np.nan,
            "success_head_available_for_selection": _success_head_is_available(df_test_new),
        }

    explore_work = _build_success_first_score_table(
        df_test_new=remaining_df,
        success_acq_success_weight=success_acq_success_weight,
        success_acq_pred_weight=success_acq_pred_weight,
        success_acq_std_penalty=success_acq_std_penalty,
        success_acq_novelty_weight=success_acq_novelty_weight,
    )

    gated_work, gate_applied, success_cutoff, pred_cutoff = _apply_success_explore_gate(
        work=explore_work,
        k=explore_k,
        success_explore_success_quantile=success_explore_success_quantile,
        success_explore_pred_quantile=success_explore_pred_quantile,
    )

    candidate_variants = gated_work["variant"].tolist()
    candidate_X = _subset_embeddings_by_variants(embeddings, candidate_variants)

    if candidate_X is None or len(candidate_X) != len(candidate_variants):
        explore_variants = (
            gated_work.sort_values(by="success_first_score", ascending=False)
            .head(explore_k)["variant"]
            .tolist()
        )
    else:
        candidate_X = candidate_X.loc[candidate_variants]
        dmat = _pairwise_distance_matrix(candidate_X)
        if dmat is None:
            explore_variants = (
                gated_work.sort_values(by="success_first_score", ascending=False)
                .head(explore_k)["variant"]
                .tolist()
            )
        else:
            n = len(gated_work)
            selected_idx = []
            remaining_idx = list(range(n))
            base_score = gated_work["success_first_score"].to_numpy(dtype=float)

            while len(selected_idx) < min(explore_k, n) and len(remaining_idx) > 0:
                if len(selected_idx) == 0:
                    best_local = int(np.argmax(base_score[remaining_idx]))
                    pick_idx = remaining_idx[best_local]
                    selected_idx.append(pick_idx)
                    remaining_idx.remove(pick_idx)
                    continue

                rem = np.array(remaining_idx, dtype=int)
                sel = np.array(selected_idx, dtype=int)
                min_dist_to_selected = dmat[np.ix_(rem, sel)].min(axis=1)
                diversity_bonus = _rank_normalize(pd.Series(min_dist_to_selected)).to_numpy(dtype=float)

                combined_score = base_score[rem] + float(redundancy_weight) * diversity_bonus
                best_local = int(np.argmax(combined_score))
                pick_idx = rem[best_local]
                selected_idx.append(int(pick_idx))
                remaining_idx.remove(int(pick_idx))

            explore_variants = gated_work.iloc[selected_idx]["variant"].tolist()

    selected = _ordered_unique(exploit_variants + explore_variants)

    if len(selected) < k:
        backfill_df = df_test_new[~df_test_new["variant"].isin(selected)].copy()
        if not backfill_df.empty:
            backfill = _select_success_topn(
                df_test_new=backfill_df,
                k=k - len(selected),
                success_acq_success_weight=success_acq_success_weight,
                success_acq_pred_weight=success_acq_pred_weight,
                success_acq_std_penalty=success_acq_std_penalty,
                success_acq_novelty_weight=0.0,
            )
            selected = _ordered_unique(selected + backfill)

    selected = selected[:k]
    exploit_count = min(len(exploit_variants), len(selected))

    return selected, {
        "selected_exploit_count": exploit_count,
        "selected_explore_count": max(0, len(selected) - exploit_count),
        "success_gate_applied": gate_applied,
        "success_gate_success_cutoff": success_cutoff,
        "success_gate_pred_cutoff": pred_cutoff,
        "success_head_available_for_selection": _success_head_is_available(df_test_new),
    }

def _select_topn(
    df_test_new: pd.DataFrame,
    k: int,
) -> List[str]:
    pred_col = _resolve_pred_col(df_test_new)
    return (
        df_test_new.sort_values(by=pred_col, ascending=False)
        .head(k)["variant"]
        .tolist()
    )


def _select_dist(
    df_test_new: pd.DataFrame,
    k: int,
) -> List[str]:
    novelty_col = _resolve_novelty_col(df_test_new)
    if novelty_col is None:
        print("Warning: learning_strategy='dist' but no novelty/distance column found. Falling back to topn.")
        return _select_topn(df_test_new, k)

    return (
        df_test_new.sort_values(by=novelty_col, ascending=False)
        .head(k)["variant"]
        .tolist()
    )


def _select_topn2bottomn2(
    df_test_new: pd.DataFrame,
    k: int,
) -> List[str]:
    pred_col = _resolve_pred_col(df_test_new)
    top_k = k // 2
    bottom_k = k - top_k

    sorted_df = df_test_new.sort_values(by=pred_col, ascending=False)
    top_half = sorted_df.head(top_k)["variant"].tolist()
    bottom_half = sorted_df.tail(bottom_k)["variant"].tolist()
    selected = top_half + bottom_half

    deduped = []
    seen = set()
    for v in selected:
        if v not in seen:
            deduped.append(v)
            seen.add(v)

    if len(deduped) < k:
        remaining = [v for v in sorted_df["variant"].tolist() if v not in seen]
        deduped.extend(remaining[: k - len(deduped)])

    return deduped[:k]



def _parse_single_mutation_position(variant: Any) -> Optional[int]:
    match = re.fullmatch(r"[A-Z](\d+)[A-Z]", str(variant).strip())
    if match is None:
        return None
    return int(match.group(1))


def _position_category(variant: Any) -> str:
    position = _parse_single_mutation_position(variant)
    if position is not None:
        return f"pos:{position}"
    return f"unparsed:{str(variant).strip()}"


def _position_entropy_stats(
    variants: List[str],
    entropy_base: str = DEFAULT_PARETO_ENTROPY_BASE,
    available_position_count: Optional[int] = None,
    normalization_batch_size: Optional[int] = None,
) -> Dict[str, float]:

    variants = [str(v) for v in variants]
    n_actual = len(variants)
    if n_actual == 0:
        return {
            "entropy": np.nan,
            "unique_positions": 0,
            "max_position_fraction": np.nan,
            "unparsed_variants": 0,
        }

    categories = [_position_category(v) for v in variants]
    counts = pd.Series(categories, dtype="object").value_counts(dropna=False)
    probs = counts.to_numpy(dtype=float) / float(n_actual)
    raw_entropy = float(-np.sum(probs * np.log(probs + 1e-15)))

    target_batch_size = int(normalization_batch_size or n_actual)
    target_batch_size = max(1, target_batch_size)
    mode = str(entropy_base or DEFAULT_PARETO_ENTROPY_BASE).strip().lower()

    if mode == "available_positions":
        available = int(available_position_count or len(counts))
        max_categories = min(target_batch_size, max(1, available))
    else:
        max_categories = target_batch_size

    if max_categories <= 1:
        normalized_entropy = 0.0
    else:
        normalized_entropy = float(raw_entropy / np.log(max_categories))
        normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))

    unparsed_count = int(
        sum(_parse_single_mutation_position(v) is None for v in variants)
    )
    return {
        "entropy": normalized_entropy,
        "unique_positions": int(len(counts)),
        "max_position_fraction": float(counts.max() / float(n_actual)),
        "unparsed_variants": unparsed_count,
    }


def _pareto_batch_objective(
    selected_variants: List[str],
    pred_norm_map: Dict[str, float],
    batch_size: int,
    lambda_value: float,
    entropy_base: str,
    available_position_count: int,
) -> Tuple[float, float, float]:
    if len(selected_variants) == 0 or batch_size <= 0:
        return -np.inf, 0.0, 0.0

    quality_sum = float(
        sum(float(pred_norm_map.get(v, 0.0)) for v in selected_variants)
    )

    quality_partial = quality_sum / float(batch_size)
    entropy_stats = _position_entropy_stats(
        selected_variants,
        entropy_base=entropy_base,
        available_position_count=available_position_count,
        normalization_batch_size=batch_size,
    )
    entropy = float(entropy_stats["entropy"])
    objective = quality_partial + float(lambda_value) * entropy
    return float(objective), float(quality_partial), entropy


def _greedy_position_entropy_batch(
    pool_df: pd.DataFrame,
    k: int,
    lambda_value: float,
    entropy_base: str,
    force_top1: bool,
) -> List[str]:
    if pool_df is None or pool_df.empty or k <= 0:
        return []

    pool = pool_df.copy().reset_index(drop=True)
    k = min(int(k), len(pool))
    variants = pool["variant"].astype(str).tolist()
    pred_norm_map = dict(
        zip(variants, pool["pareto_pred_norm"].astype(float).tolist())
    )
    available_position_count = len({_position_category(v) for v in variants})

    selected: List[str] = []
    remaining = variants.copy()

    if bool(force_top1) and remaining:
        top1 = variants[0]
        selected.append(top1)
        remaining.remove(top1)

    while len(selected) < k and remaining:
        best_variant = None
        best_key = None

        for candidate in remaining:
            trial = selected + [candidate]
            objective, quality_partial, entropy = _pareto_batch_objective(
                selected_variants=trial,
                pred_norm_map=pred_norm_map,
                batch_size=k,
                lambda_value=lambda_value,
                entropy_base=entropy_base,
                available_position_count=available_position_count,
            )

            key = (
                float(objective),
                float(pred_norm_map.get(candidate, 0.0)),
                float(entropy),
                str(candidate),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_variant = candidate

        if best_variant is None:
            break
        selected.append(best_variant)
        remaining.remove(best_variant)

    return selected[:k]


def _local_search_position_entropy_batch(
    initial_batch: List[str],
    pool_df: pd.DataFrame,
    k: int,
    lambda_value: float,
    entropy_base: str,
    force_top1: bool,
    max_steps: int,
) -> List[str]:
    if (
        not initial_batch
        or pool_df is None
        or pool_df.empty
        or int(max_steps) <= 0
    ):
        return list(initial_batch)

    pool = pool_df.copy().reset_index(drop=True)
    variants = pool["variant"].astype(str).tolist()
    pred_norm_map = dict(
        zip(variants, pool["pareto_pred_norm"].astype(float).tolist())
    )
    available_position_count = len({_position_category(v) for v in variants})
    top1 = variants[0] if variants else None

    selected = _ordered_unique([str(v) for v in initial_batch])[:k]
    current_objective, _, _ = _pareto_batch_objective(
        selected,
        pred_norm_map,
        k,
        lambda_value,
        entropy_base,
        available_position_count,
    )

    for _ in range(int(max_steps)):
        selected_set = set(selected)
        outside = [v for v in variants if v not in selected_set]
        best_trial = None
        best_key = None

        removable_indices = list(range(len(selected)))
        if bool(force_top1) and top1 in selected:
            removable_indices = [
                idx for idx in removable_indices if selected[idx] != top1
            ]

        for remove_idx in removable_indices:
            removed = selected[remove_idx]
            for incoming in outside:
                trial = selected.copy()
                trial[remove_idx] = incoming
                if len(set(trial)) != len(trial):
                    continue

                objective, _, entropy = _pareto_batch_objective(
                    trial,
                    pred_norm_map,
                    k,
                    lambda_value,
                    entropy_base,
                    available_position_count,
                )
                improvement = float(objective - current_objective)
                key = (
                    improvement,
                    float(pred_norm_map.get(incoming, 0.0)),
                    float(entropy),
                    str(incoming),
                    str(removed),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_trial = trial

        if best_key is None or best_key[0] <= 1e-12 or best_trial is None:
            break

        selected = best_trial
        current_objective += float(best_key[0])

    selected_set = set(selected)

    return [v for v in variants if v in selected_set][:k]


def _make_pareto_batch_record(
    batch: List[str],
    pool_df: pd.DataFrame,
    lambda_value: float,
    entropy_base: str,
    available_position_count: int,
) -> Dict[str, Any]:
    pred_norm_map = dict(
        zip(
            pool_df["variant"].astype(str),
            pool_df["pareto_pred_norm"].astype(float),
        )
    )
    pred_raw_map = dict(
        zip(
            pool_df["variant"].astype(str),
            pool_df["pareto_pred_raw"].astype(float),
        )
    )
    batch = _ordered_unique([str(v) for v in batch])
    stats = _position_entropy_stats(
        batch,
        entropy_base=entropy_base,
        available_position_count=available_position_count,
        normalization_batch_size=len(batch),
    )
    quality = float(np.mean([pred_norm_map[v] for v in batch])) if batch else np.nan
    mean_pred = float(np.mean([pred_raw_map[v] for v in batch])) if batch else np.nan
    return {
        "batch": batch,
        "lambda": float(lambda_value),
        "quality": quality,
        "entropy": float(stats["entropy"]),
        "mean_pred": mean_pred,
        "unique_positions": int(stats["unique_positions"]),
        "max_position_fraction": float(stats["max_position_fraction"]),
        "unparsed_variants": int(stats["unparsed_variants"]),
    }


def _filter_non_dominated_batches(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    frontier: List[Dict[str, Any]] = []
    eps = 1e-12

    for i, record_i in enumerate(records):
        dominated = False
        qi = float(record_i["quality"])
        hi = float(record_i["entropy"])

        for j, record_j in enumerate(records):
            if i == j:
                continue
            qj = float(record_j["quality"])
            hj = float(record_j["entropy"])
            weakly_better = (qj >= qi - eps) and (hj >= hi - eps)
            strictly_better = (qj > qi + eps) or (hj > hi + eps)
            if weakly_better and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(record_i)

    frontier.sort(
        key=lambda rec: (
            -float(rec["quality"]),
            float(rec["entropy"]),
            float(rec["lambda"]),
        )
    )
    return frontier


def _choose_pareto_balance_batch(
    frontier: List[Dict[str, Any]],
    topn_record: Dict[str, Any],
    balance_rule: str,
    min_quality_retention: float,
) -> Tuple[Dict[str, Any], bool]:

    if not frontier:
        return topn_record, True

    topn_quality = float(topn_record.get("quality", np.nan))
    threshold = float(np.clip(min_quality_retention, 0.0, 1.0))

    feasible = []
    for rec in frontier:
        quality = float(rec["quality"])
        if not np.isfinite(topn_quality) or topn_quality <= 1e-12:
            retention = 1.0 if quality >= topn_quality - 1e-12 else np.nan
        else:
            retention = quality / topn_quality
        rec["quality_retention"] = float(retention) if np.isfinite(retention) else np.nan
        if threshold <= 0.0 or (np.isfinite(retention) and retention >= threshold - 1e-12):
            feasible.append(rec)

    if not feasible:
        return topn_record, True

    rule = str(balance_rule or DEFAULT_PARETO_BALANCE_RULE).strip().lower()

    if rule == "max_entropy_under_quality":
        chosen = max(
            feasible,
            key=lambda rec: (
                float(rec["entropy"]),
                float(rec["quality"]),
                -float(rec["lambda"]),
            ),
        )
        return chosen, False

    if rule == "knee" and len(feasible) >= 3:
        q_values = np.array([float(rec["quality"]) for rec in feasible], dtype=float)
        h_values = np.array([float(rec["entropy"]) for rec in feasible], dtype=float)

        q_range = float(np.ptp(q_values))
        h_range = float(np.ptp(h_values))
        q_scaled = np.ones(len(feasible)) if q_range <= 1e-12 else (q_values - q_values.min()) / q_range
        h_scaled = np.ones(len(feasible)) if h_range <= 1e-12 else (h_values - h_values.min()) / h_range

        order = np.argsort(-q_scaled)
        start = np.array([q_scaled[order[0]], h_scaled[order[0]]], dtype=float)
        end = np.array([q_scaled[order[-1]], h_scaled[order[-1]]], dtype=float)
        chord = end - start
        chord_norm = float(np.linalg.norm(chord))

        if chord_norm > 1e-12:
            distances = []
            for qv, hv in zip(q_scaled, h_scaled):
                point = np.array([qv, hv], dtype=float)
                distance = abs(float(np.cross(chord, point - start))) / chord_norm
                distances.append(distance)
            best_idx = max(
                range(len(feasible)),
                key=lambda idx: (
                    float(distances[idx]),
                    float(feasible[idx]["quality"]),
                    float(feasible[idx]["entropy"]),
                ),
            )
            return feasible[best_idx], False

    chosen = max(
        feasible,
        key=lambda rec: (
            float(rec["quality"]) * float(rec["entropy"]),
            float(rec.get("quality_retention", np.nan)) if np.isfinite(rec.get("quality_retention", np.nan)) else -np.inf,
            float(rec["entropy"]),
            float(rec["quality"]),
            -float(rec["lambda"]),
        ),
    )
    return chosen, False


def _select_position_entropy_pareto(
    df_test_new: pd.DataFrame,
    k: int,
    pareto_pool_multiplier: float = DEFAULT_PARETO_POOL_MULTIPLIER,
    pareto_lambda_values: Optional[List[float]] = None,
    pareto_balance_rule: str = DEFAULT_PARETO_BALANCE_RULE,
    pareto_min_quality_retention: float = DEFAULT_PARETO_MIN_QUALITY_RETENTION,
    pareto_local_search: bool = DEFAULT_PARETO_LOCAL_SEARCH,
    pareto_local_search_steps: int = DEFAULT_PARETO_LOCAL_SEARCH_STEPS,
    pareto_entropy_base: str = DEFAULT_PARETO_ENTROPY_BASE,
    pareto_force_top1: bool = DEFAULT_PARETO_FORCE_TOP1,
) -> Tuple[List[str], Dict[str, Any]]:

    empty_meta = {
        "pareto_applied": False,
        "pareto_fallback_to_topn": False,
        "pareto_pool_size": 0,
        "pareto_front_size": 0,
        "pareto_batches_evaluated": 0,
        "pareto_selected_lambda": np.nan,
        "pareto_selected_quality": np.nan,
        "pareto_selected_entropy": np.nan,
        "pareto_selected_mean_pred": np.nan,
        "pareto_topn_quality": np.nan,
        "pareto_topn_entropy": np.nan,
        "pareto_topn_mean_pred": np.nan,
        "pareto_quality_retention": np.nan,
        "pareto_entropy_gain": np.nan,
        "pareto_selected_unique_positions": np.nan,
        "pareto_selected_max_position_fraction": np.nan,
        "pareto_topn_unique_positions": np.nan,
        "pareto_topn_max_position_fraction": np.nan,
        "pareto_unparsed_variant_count": np.nan,
        "pareto_pool_multiplier": float(pareto_pool_multiplier),
        "pareto_balance_rule": str(pareto_balance_rule),
        "pareto_min_quality_retention": float(pareto_min_quality_retention),
        "pareto_local_search": bool(pareto_local_search),
        "pareto_local_search_steps": int(pareto_local_search_steps),
        "pareto_entropy_base": str(pareto_entropy_base),
        "pareto_force_top1": bool(pareto_force_top1),
    }

    if df_test_new is None or df_test_new.empty or int(k) <= 0:
        return [], empty_meta

    work = df_test_new.copy()
    pred_col = _resolve_pred_col(work)
    work["variant"] = work["variant"].astype(str)
    work["pareto_pred_raw"] = _safe_numeric_series(work[pred_col])
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.drop_duplicates(subset=["variant"], keep="first")

    finite_pred = work["pareto_pred_raw"].dropna()
    if finite_pred.empty:
        selected = _select_topn(df_test_new, min(k, len(df_test_new)))
        meta = empty_meta.copy()
        meta["pareto_fallback_to_topn"] = True
        return selected, meta

    fill_value = float(finite_pred.min())
    work["pareto_pred_raw"] = work["pareto_pred_raw"].fillna(fill_value)
    work = work.sort_values(
        by=["pareto_pred_raw", "variant"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    k = min(int(k), len(work))
    pool_size = min(
        len(work),
        max(k, int(np.ceil(float(pareto_pool_multiplier) * k))),
    )
    pool = work.head(pool_size).copy().reset_index(drop=True)

    pred_min = float(pool["pareto_pred_raw"].min())
    pred_max = float(pool["pareto_pred_raw"].max())
    if pred_max - pred_min <= 1e-12:
        pool["pareto_pred_norm"] = 1.0
    else:
        pool["pareto_pred_norm"] = (
            pool["pareto_pred_raw"] - pred_min
        ) / (pred_max - pred_min)

    lambda_values = pareto_lambda_values
    if lambda_values is None:
        lambda_values = list(DEFAULT_PARETO_LAMBDA_VALUES)
    lambda_values = [float(value) for value in lambda_values if float(value) >= 0.0]
    lambda_values = list(dict.fromkeys(lambda_values))
    if not lambda_values:
        lambda_values = [0.0]
    if 0.0 not in lambda_values:
        lambda_values = [0.0] + lambda_values

    available_position_count = len(
        {_position_category(v) for v in pool["variant"].astype(str).tolist()}
    )
    topn_batch = pool.head(k)["variant"].astype(str).tolist()
    topn_record = _make_pareto_batch_record(
        topn_batch,
        pool,
        lambda_value=0.0,
        entropy_base=pareto_entropy_base,
        available_position_count=available_position_count,
    )

    records_by_batch: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    topn_key = tuple(sorted(topn_batch))
    records_by_batch[topn_key] = topn_record

    for lambda_value in lambda_values:
        batch = _greedy_position_entropy_batch(
            pool_df=pool,
            k=k,
            lambda_value=lambda_value,
            entropy_base=pareto_entropy_base,
            force_top1=pareto_force_top1,
        )
        if bool(pareto_local_search) and int(pareto_local_search_steps) > 0:
            batch = _local_search_position_entropy_batch(
                initial_batch=batch,
                pool_df=pool,
                k=k,
                lambda_value=lambda_value,
                entropy_base=pareto_entropy_base,
                force_top1=pareto_force_top1,
                max_steps=pareto_local_search_steps,
            )

        if len(batch) < k:
            backfill = [
                v for v in pool["variant"].astype(str).tolist() if v not in batch
            ]
            batch = _ordered_unique(batch + backfill)[:k]

        record = _make_pareto_batch_record(
            batch,
            pool,
            lambda_value=lambda_value,
            entropy_base=pareto_entropy_base,
            available_position_count=available_position_count,
        )
        key = tuple(sorted(record["batch"]))
        if key not in records_by_batch:
            records_by_batch[key] = record
        else:

            records_by_batch[key]["lambda"] = min(
                float(records_by_batch[key]["lambda"]),
                float(lambda_value),
            )

    records = list(records_by_batch.values())
    frontier = _filter_non_dominated_batches(records)
    chosen, fallback = _choose_pareto_balance_batch(
        frontier=frontier,
        topn_record=topn_record,
        balance_rule=pareto_balance_rule,
        min_quality_retention=pareto_min_quality_retention,
    )

    selected = list(chosen.get("batch", topn_batch))[:k]
    if len(selected) < k:
        selected = topn_batch
        chosen = topn_record
        fallback = True

    topn_quality = float(topn_record["quality"])
    chosen_quality = float(chosen["quality"])
    if topn_quality <= 1e-12:
        quality_retention = 1.0 if chosen_quality >= topn_quality - 1e-12 else np.nan
    else:
        quality_retention = chosen_quality / topn_quality

    overlap_with_topn = len(set(selected).intersection(topn_batch))
    meta = empty_meta.copy()
    meta.update(
        {
            "selected_exploit_count": int(overlap_with_topn),
            "selected_explore_count": int(max(0, len(selected) - overlap_with_topn)),
            "pareto_applied": True,
            "pareto_fallback_to_topn": bool(fallback),
            "pareto_pool_size": int(pool_size),
            "pareto_front_size": int(len(frontier)),
            "pareto_batches_evaluated": int(len(records)),
            "pareto_selected_lambda": float(chosen.get("lambda", np.nan)),
            "pareto_selected_quality": chosen_quality,
            "pareto_selected_entropy": float(chosen["entropy"]),
            "pareto_selected_mean_pred": float(chosen["mean_pred"]),
            "pareto_topn_quality": topn_quality,
            "pareto_topn_entropy": float(topn_record["entropy"]),
            "pareto_topn_mean_pred": float(topn_record["mean_pred"]),
            "pareto_quality_retention": float(quality_retention) if np.isfinite(quality_retention) else np.nan,
            "pareto_entropy_gain": float(chosen["entropy"] - topn_record["entropy"]),
            "pareto_selected_unique_positions": int(chosen["unique_positions"]),
            "pareto_selected_max_position_fraction": float(chosen["max_position_fraction"]),
            "pareto_topn_unique_positions": int(topn_record["unique_positions"]),
            "pareto_topn_max_position_fraction": float(topn_record["max_position_fraction"]),
            "pareto_unparsed_variant_count": int(chosen["unparsed_variants"]),
        }
    )
    return selected, meta


def _select_ucb(
    df_test_new: pd.DataFrame,
    k: int,
    ucb_beta: float,
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
) -> List[str]:
    work = _build_candidate_score_table(
        df_test_new=df_test_new,
        ucb_beta=ucb_beta,
        novelty_weight=0.0,
        success_weight=success_weight,
    )
    score_col = "acq_base_score" if float(success_weight) != 0.0 else "ucb_raw"
    return work.sort_values(by=score_col, ascending=False).head(k)["variant"].tolist()

def _select_thompson(
    df_test_new: pd.DataFrame,
    k: int,
    random_seed: int,
    thompson_scale: float,
) -> List[str]:
    pred_col = _resolve_pred_col(df_test_new)
    std_col = _resolve_std_col(df_test_new)

    work = df_test_new.copy().reset_index(drop=True)
    work["pred_mean_val"] = _safe_numeric_series(work[pred_col]).fillna(-np.inf)

    if std_col is None:
        work["pred_std_val"] = 0.0
    else:
        work["pred_std_val"] = _safe_numeric_series(work[std_col]).fillna(0.0)

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=len(work))
    work["thompson_score"] = (
        work["pred_mean_val"] + float(thompson_scale) * work["pred_std_val"] * noise
    )

    return work.sort_values(by="thompson_score", ascending=False).head(k)["variant"].tolist()


def _select_ucb_diverse(
    df_test_new: pd.DataFrame,
    embeddings: pd.DataFrame,
    k: int,
    ucb_beta: float,
    novelty_weight: float,
    redundancy_weight: float,
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
) -> List[str]:

    if df_test_new is None or df_test_new.empty:
        return []

    work = _build_candidate_score_table(
        df_test_new=df_test_new,
        ucb_beta=ucb_beta,
        novelty_weight=novelty_weight,
        success_weight=success_weight,
    )

    candidate_variants = work["variant"].tolist()
    candidate_X = _subset_embeddings_by_variants(embeddings, candidate_variants)

    if candidate_X is None or len(candidate_X) != len(candidate_variants):
        return work.sort_values(by="acq_base_score", ascending=False).head(k)["variant"].tolist()

    candidate_X = candidate_X.loc[candidate_variants]
    dmat = _pairwise_distance_matrix(candidate_X)
    if dmat is None:
        return work.sort_values(by="acq_base_score", ascending=False).head(k)["variant"].tolist()

    n = len(work)
    selected_idx = []
    remaining_idx = list(range(n))

    base_score = work["acq_base_score"].to_numpy(dtype=float)

    while len(selected_idx) < min(k, n) and len(remaining_idx) > 0:
        if len(selected_idx) == 0:
            best_local = int(np.argmax(base_score[remaining_idx]))
            pick_idx = remaining_idx[best_local]
            selected_idx.append(pick_idx)
            remaining_idx.remove(pick_idx)
            continue

        rem = np.array(remaining_idx, dtype=int)
        sel = np.array(selected_idx, dtype=int)
        min_dist_to_selected = dmat[np.ix_(rem, sel)].min(axis=1)

        diversity_bonus = _rank_normalize(pd.Series(min_dist_to_selected)).to_numpy(dtype=float)

        combined_score = base_score[rem] + float(redundancy_weight) * diversity_bonus
        best_local = int(np.argmax(combined_score))
        pick_idx = rem[best_local]

        selected_idx.append(int(pick_idx))
        remaining_idx.remove(int(pick_idx))

    return work.iloc[selected_idx]["variant"].tolist()

def _select_mixed_ucb_diverse(
    df_test_new: pd.DataFrame,
    embeddings: pd.DataFrame,
    k: int,
    ucb_beta: float,
    novelty_weight: float,
    redundancy_weight: float,
    mixed_exploit_frac: float,
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
) -> Tuple[List[str], Dict[str, Any]]:

    if df_test_new is None or df_test_new.empty:
        return [], {
            "selected_exploit_count": 0,
            "selected_explore_count": 0,
        }

    n_available = len(df_test_new)
    k = min(k, n_available)

    exploit_k, explore_k = _split_mixed_counts(
        k=k,
        mixed_exploit_frac=mixed_exploit_frac,
    )

    exploit_variants = _select_topn(df_test_new, exploit_k)

    remaining_df = df_test_new[~df_test_new["variant"].isin(exploit_variants)].copy()

    if explore_k <= 0 or remaining_df.empty:
        selected = exploit_variants[:k]
        return selected, {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
        }

    explore_variants = _select_ucb_diverse(
        df_test_new=remaining_df,
        embeddings=embeddings,
        k=explore_k,
        ucb_beta=ucb_beta,
        novelty_weight=novelty_weight,
        redundancy_weight=redundancy_weight,
        success_weight=success_weight,
    )

    selected = _ordered_unique(exploit_variants + explore_variants)

    if len(selected) < k:
        backfill_df = df_test_new[~df_test_new["variant"].isin(selected)].copy()
        if not backfill_df.empty:
            backfill = _select_topn(backfill_df, k - len(selected))
            selected = _ordered_unique(selected + backfill)

    selected = selected[:k]

    return selected, {
        "selected_exploit_count": min(len(exploit_variants), len(selected)),
        "selected_explore_count": max(0, len(selected) - min(len(exploit_variants), len(selected))),
    }


def _build_safe_score_table(
    df_test_new: pd.DataFrame,
    safe_std_penalty: float = DEFAULT_SAFE_STD_PENALTY,
    safe_novelty_weight: float = DEFAULT_SAFE_NOVELTY_WEIGHT,
) -> pd.DataFrame:

    if df_test_new is None or df_test_new.empty:
        return pd.DataFrame()

    pred_col = _resolve_pred_col(df_test_new)
    std_col = _resolve_std_col(df_test_new)
    novelty_col = _resolve_novelty_col(df_test_new)

    work = df_test_new.copy().reset_index(drop=True)
    work["pred_mean_val"] = _safe_numeric_series(work[pred_col]).fillna(-np.inf)

    if std_col is not None:
        work["pred_std_val"] = _safe_numeric_series(work[std_col]).fillna(0.0)
    else:
        work["pred_std_val"] = 0.0

    if novelty_col is not None:
        work["novelty_val"] = _safe_numeric_series(work[novelty_col]).fillna(0.0)
    else:
        work["novelty_val"] = 0.0

    work["pred_rank"] = _rank_normalize(work["pred_mean_val"])
    work["std_rank"] = _rank_normalize(work["pred_std_val"])
    work["novelty_rank"] = _rank_normalize(work["novelty_val"])

    work["safe_score"] = (
        work["pred_rank"]
        - float(safe_std_penalty) * work["std_rank"]
        + float(safe_novelty_weight) * work["novelty_rank"]
    )
    work["safe_score_base"] = work["safe_score"]

    return work


def _batch_mean_score(
    score_table: pd.DataFrame,
    variants: List[str],
    score_col: str = "safe_score",
) -> float:

    if score_table is None or score_table.empty or not variants:
        return np.nan
    if score_col not in score_table.columns:
        return np.nan
    vals = score_table.loc[score_table["variant"].isin(variants), score_col]
    vals = _safe_numeric_series(vals)
    return float(vals.mean()) if vals.notna().any() else np.nan


def _select_safe_struct_mixed(
    df_test_new: pd.DataFrame,
    embeddings: Optional[pd.DataFrame],
    k: int,
    *,
    safe_max_replace: int = DEFAULT_SAFE_MAX_REPLACE,
    safe_candidate_pool_quantile: float = DEFAULT_SAFE_CANDIDATE_POOL_QUANTILE,
    safe_std_penalty: float = DEFAULT_SAFE_STD_PENALTY,
    safe_novelty_weight: float = DEFAULT_SAFE_NOVELTY_WEIGHT,
    safe_batch_margin: float = DEFAULT_SAFE_BATCH_MARGIN,
    safe_require_batch_score: bool = DEFAULT_SAFE_REQUIRE_BATCH_SCORE,
    redundancy_weight: float = DEFAULT_REDUNDANCY_WEIGHT,
) -> Tuple[List[str], Dict[str, Any]]:

    empty_meta = {
        "selected_exploit_count": 0,
        "selected_explore_count": 0,
        "safe_guard_applied": False,
        "safe_fallback_to_topn": False,
        "safe_candidate_pool_size": 0,
        "safe_replaced_count": 0,
        "safe_topn_batch_score": np.nan,
        "safe_candidate_batch_score": np.nan,
        "safe_candidate_pool_quantile": float(safe_candidate_pool_quantile),
        "safe_max_replace": int(safe_max_replace),
        "safe_std_penalty": float(safe_std_penalty),
        "safe_novelty_weight": float(safe_novelty_weight),
        "safe_batch_margin": float(safe_batch_margin),
        "safe_require_batch_score": bool(safe_require_batch_score),
    }

    if df_test_new is None or df_test_new.empty:
        return [], empty_meta

    n_available = len(df_test_new)
    k = min(int(k), n_available)
    if k <= 0:
        return [], empty_meta

    topn_variants = _select_topn(df_test_new, k)
    if len(topn_variants) <= 1:
        meta = empty_meta.copy()
        meta.update(
            {
                "selected_exploit_count": len(topn_variants),
                "selected_explore_count": 0,
                "safe_topn_batch_score": np.nan,
                "safe_candidate_batch_score": np.nan,
            }
        )
        return topn_variants, meta

    work = _build_safe_score_table(
        df_test_new=df_test_new,
        safe_std_penalty=safe_std_penalty,
        safe_novelty_weight=safe_novelty_weight,
    )

    topn_score = _batch_mean_score(work, topn_variants, "safe_score")

    max_replace = int(max(0, safe_max_replace))
    max_replace = min(max_replace, max(0, k - 1))
    if max_replace <= 0:
        meta = empty_meta.copy()
        meta.update(
            {
                "selected_exploit_count": len(topn_variants),
                "selected_explore_count": 0,
                "safe_topn_batch_score": topn_score,
                "safe_candidate_batch_score": topn_score,
            }
        )
        return topn_variants, meta

    protected_count = max(1, k - max_replace)
    protected_variants = topn_variants[:protected_count]
    replaceable_variants = topn_variants[protected_count:]

    q = float(min(0.99, max(0.0, safe_candidate_pool_quantile)))
    pred_vals = _safe_numeric_series(work["pred_mean_val"])
    pred_cutoff = float(pred_vals.quantile(q)) if pred_vals.notna().any() else -np.inf

    candidate_pool = work[
        (~work["variant"].isin(protected_variants))
        & (_safe_numeric_series(work["pred_mean_val"]) >= pred_cutoff)
    ].copy()

    if len(candidate_pool) < max_replace:
        candidate_pool = work[~work["variant"].isin(protected_variants)].copy()

    candidate_pool = candidate_pool.reset_index(drop=True)
    candidate_pool_size = int(len(candidate_pool))

    if candidate_pool.empty:
        meta = empty_meta.copy()
        meta.update(
            {
                "selected_exploit_count": len(topn_variants),
                "selected_explore_count": 0,
                "safe_fallback_to_topn": True,
                "safe_candidate_pool_size": 0,
                "safe_topn_batch_score": topn_score,
                "safe_candidate_batch_score": topn_score,
            }
        )
        return topn_variants, meta

    candidate_variants = candidate_pool["variant"].tolist()
    selected_replacements: List[str] = []

    candidate_X = _subset_embeddings_by_variants(embeddings, candidate_variants) if embeddings is not None else None

    if candidate_X is None or len(candidate_X) != len(candidate_variants):
        selected_replacements = (
            candidate_pool.sort_values(by="safe_score", ascending=False)
            .head(max_replace)["variant"]
            .tolist()
        )
    else:
        candidate_X = candidate_X.loc[candidate_variants]
        selected_context = list(protected_variants)
        remaining = candidate_pool.copy().reset_index(drop=True)

        while len(selected_replacements) < max_replace and not remaining.empty:
            scores = _safe_numeric_series(remaining["safe_score"]).to_numpy(dtype=float)

            diversity_bonus = np.zeros(len(remaining), dtype=float)
            if selected_context:
                X_remaining = _subset_embeddings_by_variants(embeddings, remaining["variant"].tolist())
                X_context = _subset_embeddings_by_variants(embeddings, selected_context)
                if X_remaining is not None and X_context is not None and len(X_remaining) == len(remaining):
                    X_remaining = X_remaining.loc[remaining["variant"].tolist()]
                    X_context = X_context.loc[[v for v in selected_context if v in X_context.index]]
                    if len(X_context) > 0:
                        d = cdist(X_remaining.values, X_context.values, metric="euclidean").min(axis=1)
                        diversity_bonus = _rank_normalize(pd.Series(d)).to_numpy(dtype=float)

            combined = scores + float(redundancy_weight) * diversity_bonus
            pick_pos = int(np.nanargmax(combined))
            pick_variant = str(remaining.iloc[pick_pos]["variant"])
            selected_replacements.append(pick_variant)
            selected_context.append(pick_variant)
            remaining = remaining[remaining["variant"] != pick_variant].reset_index(drop=True)

    candidate_batch = _ordered_unique(protected_variants + selected_replacements)

    if len(candidate_batch) < k:
        candidate_batch = _ordered_unique(candidate_batch + replaceable_variants)

    if len(candidate_batch) < k:
        backfill = [
            v for v in _select_topn(df_test_new[~df_test_new["variant"].isin(candidate_batch)].copy(), k - len(candidate_batch))
            if v not in candidate_batch
        ]
        candidate_batch = _ordered_unique(candidate_batch + backfill)

    candidate_batch = candidate_batch[:k]
    candidate_score = _batch_mean_score(work, candidate_batch, "safe_score")

    fallback = False
    if bool(safe_require_batch_score):
        if pd.isna(candidate_score) or pd.isna(topn_score):
            fallback = True
        elif candidate_score < (topn_score - float(safe_batch_margin)):
            fallback = True

    if fallback:
        selected = topn_variants
        final_candidate_score = topn_score
        replaced_count = 0
    else:
        selected = candidate_batch
        final_candidate_score = candidate_score
        replaced_count = len([v for v in selected if v not in topn_variants])

    selected = selected[:k]

    meta = empty_meta.copy()
    meta.update(
        {
            "selected_exploit_count": max(0, k - int(replaced_count)),
            "selected_explore_count": int(replaced_count),
            "safe_guard_applied": True,
            "safe_fallback_to_topn": bool(fallback),
            "safe_candidate_pool_size": candidate_pool_size,
            "safe_replaced_count": int(replaced_count),
            "safe_topn_batch_score": topn_score,
            "safe_candidate_batch_score": final_candidate_score,
            "safe_candidate_pool_quantile": float(safe_candidate_pool_quantile),
            "safe_max_replace": int(safe_max_replace),
            "safe_std_penalty": float(safe_std_penalty),
            "safe_novelty_weight": float(safe_novelty_weight),
            "safe_batch_margin": float(safe_batch_margin),
            "safe_require_batch_score": bool(safe_require_batch_score),
        }
    )

    return selected, meta


def _select_next_round_variants(
    df_test_new: pd.DataFrame,
    learning_strategy: str,
    num_mutants_per_round: int,
    random_seed: int,
    embeddings: Optional[pd.DataFrame] = None,
    ucb_beta: float = DEFAULT_UCB_BETA,
    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT,
    redundancy_weight: float = DEFAULT_REDUNDANCY_WEIGHT,
    thompson_scale: float = DEFAULT_THOMPSON_SCALE,
    mixed_exploit_frac: float = DEFAULT_MIXED_EXPLOIT_FRAC,
    current_round: Optional[int] = None,
    mixed_exploit_schedule: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]] = None,
    parsed_mixed_schedule_map: Optional[Dict[int, float]] = None,
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
    success_filter_quantile: float = DEFAULT_SUCCESS_FILTER_QUANTILE,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT,
    success_explore_success_quantile: float = DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE,
    success_explore_pred_quantile: float = DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE,
    safe_max_replace: int = DEFAULT_SAFE_MAX_REPLACE,
    safe_candidate_pool_quantile: float = DEFAULT_SAFE_CANDIDATE_POOL_QUANTILE,
    safe_std_penalty: float = DEFAULT_SAFE_STD_PENALTY,
    safe_novelty_weight: float = DEFAULT_SAFE_NOVELTY_WEIGHT,
    safe_batch_margin: float = DEFAULT_SAFE_BATCH_MARGIN,
    safe_require_batch_score: bool = DEFAULT_SAFE_REQUIRE_BATCH_SCORE,
    pareto_pool_multiplier: float = DEFAULT_PARETO_POOL_MULTIPLIER,
    pareto_lambda_values: Optional[List[float]] = None,
    pareto_balance_rule: str = DEFAULT_PARETO_BALANCE_RULE,
    pareto_min_quality_retention: float = DEFAULT_PARETO_MIN_QUALITY_RETENTION,
    pareto_local_search: bool = DEFAULT_PARETO_LOCAL_SEARCH,
    pareto_local_search_steps: int = DEFAULT_PARETO_LOCAL_SEARCH_STEPS,
    pareto_entropy_base: str = DEFAULT_PARETO_ENTROPY_BASE,
    pareto_force_top1: bool = DEFAULT_PARETO_FORCE_TOP1,
) -> Tuple[List[str], Dict[str, Any]]:

    if df_test_new is None or df_test_new.empty:
        return [], {
            "selected_exploit_count": 0,
            "selected_explore_count": 0,
            "effective_mixed_exploit_frac": np.nan,
            "mixed_schedule_applied": False,
            "success_filter_applied": False,
            "success_mode": _normalize_success_mode(success_mode),
            "success_weight": float(success_weight),
            "success_filter_quantile": float(success_filter_quantile),
            "success_gate_applied": False,
            "success_gate_success_cutoff": np.nan,
            "success_gate_pred_cutoff": np.nan,
            "success_head_available_for_selection": False,
            "success_acq_success_weight": float(success_acq_success_weight),
            "success_acq_pred_weight": float(success_acq_pred_weight),
            "success_acq_std_penalty": float(success_acq_std_penalty),
            "success_acq_novelty_weight": float(success_acq_novelty_weight),
            "success_explore_success_quantile": float(success_explore_success_quantile),
            "success_explore_pred_quantile": float(success_explore_pred_quantile),
            "safe_guard_applied": False,
            "safe_fallback_to_topn": False,
            "safe_candidate_pool_size": 0,
            "safe_replaced_count": 0,
            "safe_topn_batch_score": np.nan,
            "safe_candidate_batch_score": np.nan,
            "safe_max_replace": int(safe_max_replace),
            "safe_candidate_pool_quantile": float(safe_candidate_pool_quantile),
            "safe_std_penalty": float(safe_std_penalty),
            "safe_novelty_weight": float(safe_novelty_weight),
            "safe_batch_margin": float(safe_batch_margin),
            "safe_require_batch_score": bool(safe_require_batch_score),
            "pareto_applied": False,
            "pareto_fallback_to_topn": False,
            "pareto_pool_size": 0,
            "pareto_front_size": 0,
            "pareto_batches_evaluated": 0,
            "pareto_selected_lambda": np.nan,
            "pareto_selected_quality": np.nan,
            "pareto_selected_entropy": np.nan,
            "pareto_selected_mean_pred": np.nan,
            "pareto_topn_quality": np.nan,
            "pareto_topn_entropy": np.nan,
            "pareto_topn_mean_pred": np.nan,
            "pareto_quality_retention": np.nan,
            "pareto_entropy_gain": np.nan,
            "pareto_selected_unique_positions": np.nan,
            "pareto_selected_max_position_fraction": np.nan,
            "pareto_topn_unique_positions": np.nan,
            "pareto_topn_max_position_fraction": np.nan,
            "pareto_unparsed_variant_count": np.nan,
            "pareto_pool_multiplier": float(pareto_pool_multiplier),
            "pareto_balance_rule": str(pareto_balance_rule),
            "pareto_min_quality_retention": float(pareto_min_quality_retention),
            "pareto_local_search": bool(pareto_local_search),
            "pareto_local_search_steps": int(pareto_local_search_steps),
            "pareto_entropy_base": str(pareto_entropy_base),
            "pareto_force_top1": bool(pareto_force_top1),
        }

    original_n_available = len(df_test_new)
    k = min(num_mutants_per_round, original_n_available)

    mode = _normalize_success_mode(success_mode)
    filtered_df = _apply_success_filter_to_candidates(
        df_test_new=df_test_new,
        k=k,
        success_filter_quantile=success_filter_quantile,
        success_mode=mode,
    )
    success_filter_applied = bool(len(filtered_df) < len(df_test_new))
    df_for_select = filtered_df

    n_available = len(df_for_select)
    k = min(num_mutants_per_round, n_available)

    effective_mixed_exploit_frac = _resolve_mixed_exploit_frac_for_round(
        round_num=current_round if current_round is not None else 1,
        mixed_exploit_frac=mixed_exploit_frac,
        mixed_exploit_schedule=mixed_exploit_schedule,
        parsed_schedule_map=parsed_mixed_schedule_map,
    )
    mixed_schedule_applied = (
        learning_strategy in {"mixed_ucb_diverse", "mixed_ucb_diverse_schedule", "success_constrained_mixed", "safe_struct_mixed", "guarded_struct_mixed"}
        and parsed_mixed_schedule_map is not None
        and current_round is not None
        and int(current_round) in parsed_mixed_schedule_map
    )

    effective_success_weight = float(success_weight) if _success_head_is_available(df_for_select) else 0.0

    if learning_strategy == "dist":
        selected = _select_dist(df_for_select, k)
        meta = {
            "selected_exploit_count": 0,
            "selected_explore_count": len(selected),
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy == "random":
        selected = _safe_pick_random_variants(
            variants=df_for_select["variant"].tolist(),
            k=k,
            seed=random_seed,
        )
        meta = {
            "selected_exploit_count": 0,
            "selected_explore_count": len(selected),
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy == "topn":
        selected = _select_topn(df_for_select, k)
        meta = {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy == "topn2bottomn2":
        selected = _select_topn2bottomn2(df_for_select, k)
        meta = {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy == "ucb":
        selected = _select_ucb(
            df_test_new=df_for_select,
            k=k,
            ucb_beta=ucb_beta,
            success_weight=effective_success_weight,
        )
        meta = {
            "selected_exploit_count": 0,
            "selected_explore_count": len(selected),
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy == "ucb_diverse":
        selected = _select_ucb_diverse(
            df_test_new=df_for_select,
            embeddings=embeddings,
            k=k,
            ucb_beta=ucb_beta,
            novelty_weight=novelty_weight,
            redundancy_weight=redundancy_weight,
            success_weight=effective_success_weight,
        )
        meta = {
            "selected_exploit_count": 0,
            "selected_explore_count": len(selected),
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    elif learning_strategy in {"mixed_ucb_diverse", "mixed_ucb_diverse_schedule"}:
        selected, meta = _select_mixed_ucb_diverse(
            df_test_new=df_for_select,
            embeddings=embeddings,
            k=k,
            ucb_beta=ucb_beta,
            novelty_weight=novelty_weight,
            redundancy_weight=redundancy_weight,
            mixed_exploit_frac=effective_mixed_exploit_frac,
            success_weight=effective_success_weight,
        )
        meta["effective_mixed_exploit_frac"] = effective_mixed_exploit_frac
        meta["mixed_schedule_applied"] = mixed_schedule_applied

    elif learning_strategy == "success_topn":
        selected = _select_success_topn(
            df_test_new=df_for_select,
            k=k,
            success_acq_success_weight=success_acq_success_weight,
            success_acq_pred_weight=success_acq_pred_weight,
            success_acq_std_penalty=success_acq_std_penalty,
            success_acq_novelty_weight=0.0,
        )
        meta = {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
            "success_gate_applied": False,
            "success_gate_success_cutoff": np.nan,
            "success_gate_pred_cutoff": np.nan,
            "success_head_available_for_selection": _success_head_is_available(df_for_select),
        }

    elif learning_strategy == "success_constrained_mixed":
        selected, meta = _select_success_constrained_mixed(
            df_test_new=df_for_select,
            embeddings=embeddings,
            k=k,
            mixed_exploit_frac=effective_mixed_exploit_frac,
            success_acq_success_weight=success_acq_success_weight,
            success_acq_pred_weight=success_acq_pred_weight,
            success_acq_std_penalty=success_acq_std_penalty,
            success_acq_novelty_weight=success_acq_novelty_weight,
            success_explore_success_quantile=success_explore_success_quantile,
            success_explore_pred_quantile=success_explore_pred_quantile,
            redundancy_weight=redundancy_weight,
        )
        meta["effective_mixed_exploit_frac"] = effective_mixed_exploit_frac
        meta["mixed_schedule_applied"] = mixed_schedule_applied

    elif learning_strategy in {"safe_struct_mixed", "guarded_struct_mixed"}:
        selected, meta = _select_safe_struct_mixed(
            df_test_new=df_for_select,
            embeddings=embeddings,
            k=k,
            safe_max_replace=safe_max_replace,
            safe_candidate_pool_quantile=safe_candidate_pool_quantile,
            safe_std_penalty=safe_std_penalty,
            safe_novelty_weight=safe_novelty_weight,
            safe_batch_margin=safe_batch_margin,
            safe_require_batch_score=safe_require_batch_score,
            redundancy_weight=redundancy_weight,
        )
        meta["effective_mixed_exploit_frac"] = effective_mixed_exploit_frac
        meta["mixed_schedule_applied"] = mixed_schedule_applied

    elif learning_strategy == "position_entropy_pareto":
        selected, meta = _select_position_entropy_pareto(
            df_test_new=df_for_select,
            k=k,
            pareto_pool_multiplier=pareto_pool_multiplier,
            pareto_lambda_values=pareto_lambda_values,
            pareto_balance_rule=pareto_balance_rule,
            pareto_min_quality_retention=pareto_min_quality_retention,
            pareto_local_search=pareto_local_search,
            pareto_local_search_steps=pareto_local_search_steps,
            pareto_entropy_base=pareto_entropy_base,
            pareto_force_top1=pareto_force_top1,
        )
        meta["effective_mixed_exploit_frac"] = effective_mixed_exploit_frac
        meta["mixed_schedule_applied"] = mixed_schedule_applied

    elif learning_strategy == "thompson":
        selected = _select_thompson(
            df_test_new=df_for_select,
            k=k,
            random_seed=random_seed,
            thompson_scale=thompson_scale,
        )
        meta = {
            "selected_exploit_count": 0,
            "selected_explore_count": len(selected),
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    else:
        print(f"Warning: unknown learning_strategy='{learning_strategy}'. Falling back to topn.")
        selected = _select_topn(df_for_select, k)
        meta = {
            "selected_exploit_count": len(selected),
            "selected_explore_count": 0,
            "effective_mixed_exploit_frac": effective_mixed_exploit_frac,
            "mixed_schedule_applied": mixed_schedule_applied,
        }

    meta["success_filter_applied"] = success_filter_applied
    meta["success_mode"] = mode
    meta["success_weight"] = effective_success_weight
    meta["success_filter_quantile"] = float(success_filter_quantile)
    meta["success_candidates_before_filter"] = int(original_n_available)
    meta["success_candidates_after_filter"] = int(n_available)
    meta["success_gate_applied"] = meta.get("success_gate_applied", False)
    meta["success_gate_success_cutoff"] = meta.get("success_gate_success_cutoff", np.nan)
    meta["success_gate_pred_cutoff"] = meta.get("success_gate_pred_cutoff", np.nan)
    meta["success_head_available_for_selection"] = meta.get("success_head_available_for_selection", _success_head_is_available(df_for_select))
    meta["success_acq_success_weight"] = float(success_acq_success_weight)
    meta["success_acq_pred_weight"] = float(success_acq_pred_weight)
    meta["success_acq_std_penalty"] = float(success_acq_std_penalty)
    meta["success_acq_novelty_weight"] = float(success_acq_novelty_weight)
    meta["success_explore_success_quantile"] = float(success_explore_success_quantile)
    meta["success_explore_pred_quantile"] = float(success_explore_pred_quantile)
    meta["safe_guard_applied"] = meta.get("safe_guard_applied", False)
    meta["safe_fallback_to_topn"] = meta.get("safe_fallback_to_topn", False)
    meta["safe_candidate_pool_size"] = meta.get("safe_candidate_pool_size", np.nan)
    meta["safe_replaced_count"] = meta.get("safe_replaced_count", np.nan)
    meta["safe_topn_batch_score"] = meta.get("safe_topn_batch_score", np.nan)
    meta["safe_candidate_batch_score"] = meta.get("safe_candidate_batch_score", np.nan)
    meta["safe_max_replace"] = int(meta.get("safe_max_replace", safe_max_replace))
    meta["safe_candidate_pool_quantile"] = float(meta.get("safe_candidate_pool_quantile", safe_candidate_pool_quantile))
    meta["safe_std_penalty"] = float(meta.get("safe_std_penalty", safe_std_penalty))
    meta["safe_novelty_weight"] = float(meta.get("safe_novelty_weight", safe_novelty_weight))
    meta["safe_batch_margin"] = float(meta.get("safe_batch_margin", safe_batch_margin))
    meta["safe_require_batch_score"] = bool(meta.get("safe_require_batch_score", safe_require_batch_score))

    meta["pareto_applied"] = bool(meta.get("pareto_applied", False))
    meta["pareto_fallback_to_topn"] = bool(meta.get("pareto_fallback_to_topn", False))
    meta["pareto_pool_size"] = meta.get("pareto_pool_size", np.nan)
    meta["pareto_front_size"] = meta.get("pareto_front_size", np.nan)
    meta["pareto_batches_evaluated"] = meta.get("pareto_batches_evaluated", np.nan)
    meta["pareto_selected_lambda"] = meta.get("pareto_selected_lambda", np.nan)
    meta["pareto_selected_quality"] = meta.get("pareto_selected_quality", np.nan)
    meta["pareto_selected_entropy"] = meta.get("pareto_selected_entropy", np.nan)
    meta["pareto_selected_mean_pred"] = meta.get("pareto_selected_mean_pred", np.nan)
    meta["pareto_topn_quality"] = meta.get("pareto_topn_quality", np.nan)
    meta["pareto_topn_entropy"] = meta.get("pareto_topn_entropy", np.nan)
    meta["pareto_topn_mean_pred"] = meta.get("pareto_topn_mean_pred", np.nan)
    meta["pareto_quality_retention"] = meta.get("pareto_quality_retention", np.nan)
    meta["pareto_entropy_gain"] = meta.get("pareto_entropy_gain", np.nan)
    meta["pareto_selected_unique_positions"] = meta.get("pareto_selected_unique_positions", np.nan)
    meta["pareto_selected_max_position_fraction"] = meta.get("pareto_selected_max_position_fraction", np.nan)
    meta["pareto_topn_unique_positions"] = meta.get("pareto_topn_unique_positions", np.nan)
    meta["pareto_topn_max_position_fraction"] = meta.get("pareto_topn_max_position_fraction", np.nan)
    meta["pareto_unparsed_variant_count"] = meta.get("pareto_unparsed_variant_count", np.nan)
    meta["pareto_pool_multiplier"] = float(meta.get("pareto_pool_multiplier", pareto_pool_multiplier))
    meta["pareto_balance_rule"] = str(meta.get("pareto_balance_rule", pareto_balance_rule))
    meta["pareto_min_quality_retention"] = float(meta.get("pareto_min_quality_retention", pareto_min_quality_retention))
    meta["pareto_local_search"] = bool(meta.get("pareto_local_search", pareto_local_search))
    meta["pareto_local_search_steps"] = int(meta.get("pareto_local_search_steps", pareto_local_search_steps))
    meta["pareto_entropy_base"] = str(meta.get("pareto_entropy_base", pareto_entropy_base))
    meta["pareto_force_top1"] = bool(meta.get("pareto_force_top1", pareto_force_top1))

    return selected, meta

def _summarize_selected_batch(
    df_test_new: pd.DataFrame,
    selected_variants: List[str],
    embeddings: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:

    empty_result = {
        "avg_selected_pred_mean": np.nan,
        "avg_selected_pred_mean_base": np.nan,
        "avg_selected_pred_std": np.nan,
        "avg_selected_novelty_to_tested": np.nan,
        "avg_selected_ucb_score_base": np.nan,
        "avg_selected_pairwise_distance": np.nan,
        "avg_selected_min_pairwise_distance": np.nan,
        "avg_selected_head_rerank_score": np.nan,
        "avg_selected_head_rerank_correction": np.nan,
        "frac_selected_head_rerank_applied": np.nan,
        "avg_selected_success_prob": np.nan,
        "min_selected_success_prob": np.nan,
        "avg_selected_success_prob_std": np.nan,
        "avg_selected_success_score_base": np.nan,
        "frac_selected_success_head_enabled": np.nan,
        "selected_position_entropy": np.nan,
        "selected_unique_positions": np.nan,
        "selected_max_position_fraction": np.nan,
        "selected_unparsed_variant_count": np.nan,
    }

    if df_test_new is None or len(selected_variants) == 0:
        return empty_result

    work = df_test_new.copy()
    pred_col = _resolve_pred_col(work)
    base_pred_col = _resolve_base_pred_col(work)
    std_col = _resolve_std_col(work)
    novelty_col = _resolve_novelty_col(work)
    success_col = _resolve_success_prob_col(work)
    success_std_col = _resolve_success_prob_std_col(work)

    selected_df = work[work["variant"].isin(selected_variants)].copy()
    if selected_df.empty:
        return empty_result

    selected_df["variant"] = pd.Categorical(
        selected_df["variant"],
        categories=selected_variants,
        ordered=True,
    )
    selected_df = selected_df.sort_values("variant").reset_index(drop=True)

    pred_mean_vals = _safe_numeric_series(selected_df[pred_col])
    if base_pred_col is not None:
        pred_mean_base_vals = _safe_numeric_series(selected_df[base_pred_col])
    else:
        pred_mean_base_vals = pred_mean_vals.copy()

    pred_std_vals = _safe_numeric_series(selected_df[std_col]) if std_col is not None else pd.Series(np.zeros(len(selected_df)))
    novelty_vals = _safe_numeric_series(selected_df[novelty_col]) if novelty_col is not None else pd.Series(np.zeros(len(selected_df)))

    if "ucb_score_base" in selected_df.columns:
        ucb_vals = _safe_numeric_series(selected_df["ucb_score_base"])
    else:
        ucb_vals = pred_mean_vals

    if "head_rerank_score" in selected_df.columns:
        head_rerank_score_vals = _safe_numeric_series(selected_df["head_rerank_score"])
    else:
        head_rerank_score_vals = pd.Series(np.zeros(len(selected_df)))

    if "head_rerank_correction" in selected_df.columns:
        head_rerank_correction_vals = _safe_numeric_series(selected_df["head_rerank_correction"])
    else:
        head_rerank_correction_vals = pd.Series(np.zeros(len(selected_df)))

    if "head_rerank_applied" in selected_df.columns:
        head_rerank_applied_vals = _safe_numeric_series(selected_df["head_rerank_applied"]).fillna(0.0)
    else:
        head_rerank_applied_vals = pd.Series(np.zeros(len(selected_df)))

    if success_col is not None:
        success_prob_vals = _safe_numeric_series(selected_df[success_col])
    else:
        success_prob_vals = pd.Series(np.full(len(selected_df), np.nan))

    if success_std_col is not None:
        success_prob_std_vals = _safe_numeric_series(selected_df[success_std_col])
    else:
        success_prob_std_vals = pd.Series(np.full(len(selected_df), np.nan))

    if "success_score_base" in selected_df.columns:
        success_score_vals = _safe_numeric_series(selected_df["success_score_base"])
    else:
        success_score_vals = success_prob_vals.copy()

    if "success_head_enabled" in selected_df.columns:
        success_enabled_vals = _safe_numeric_series(selected_df["success_head_enabled"]).fillna(0.0)
    else:
        success_enabled_vals = pd.Series(np.zeros(len(selected_df)))

    pairwise_avg = np.nan
    min_pairwise_avg = np.nan

    if embeddings is not None:
        X_sel = _subset_embeddings_by_variants(embeddings, selected_variants)
        if X_sel is not None and len(X_sel) == len(selected_variants):
            X_sel = X_sel.loc[selected_variants]
            pairwise_avg = _avg_pairwise_distance(X_sel)
            min_pairwise_avg = _avg_min_pairwise_distance(X_sel)

    position_stats = _position_entropy_stats(
        selected_variants,
        entropy_base="batch",
        available_position_count=None,
        normalization_batch_size=len(selected_variants),
    )

    return {
        "avg_selected_pred_mean": float(pred_mean_vals.mean()) if len(pred_mean_vals) > 0 else np.nan,
        "avg_selected_pred_mean_base": float(pred_mean_base_vals.mean()) if len(pred_mean_base_vals) > 0 else np.nan,
        "avg_selected_pred_std": float(pred_std_vals.mean()) if len(pred_std_vals) > 0 else np.nan,
        "avg_selected_novelty_to_tested": float(novelty_vals.mean()) if len(novelty_vals) > 0 else np.nan,
        "avg_selected_ucb_score_base": float(ucb_vals.mean()) if len(ucb_vals) > 0 else np.nan,
        "avg_selected_pairwise_distance": pairwise_avg,
        "avg_selected_min_pairwise_distance": min_pairwise_avg,
        "avg_selected_head_rerank_score": float(head_rerank_score_vals.mean()) if len(head_rerank_score_vals) > 0 else np.nan,
        "avg_selected_head_rerank_correction": float(head_rerank_correction_vals.mean()) if len(head_rerank_correction_vals) > 0 else np.nan,
        "frac_selected_head_rerank_applied": float(head_rerank_applied_vals.mean()) if len(head_rerank_applied_vals) > 0 else np.nan,
        "avg_selected_success_prob": float(success_prob_vals.mean()) if success_prob_vals.notna().any() else np.nan,
        "min_selected_success_prob": float(success_prob_vals.min()) if success_prob_vals.notna().any() else np.nan,
        "avg_selected_success_prob_std": float(success_prob_std_vals.mean()) if success_prob_std_vals.notna().any() else np.nan,
        "avg_selected_success_score_base": float(success_score_vals.mean()) if success_score_vals.notna().any() else np.nan,
        "frac_selected_success_head_enabled": float(success_enabled_vals.mean()) if len(success_enabled_vals) > 0 else np.nan,
        "selected_position_entropy": float(position_stats["entropy"]),
        "selected_unique_positions": int(position_stats["unique_positions"]),
        "selected_max_position_fraction": float(position_stats["max_position_fraction"]),
        "selected_unparsed_variant_count": int(position_stats["unparsed_variants"]),
    }

def directed_evolution_simulation(
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    num_simulations: int,
    num_iterations: int,
    num_mutants_per_round: int = 10,
    measured_var: str = "activity",
    regression_type: str = "ridge",
    learning_strategy: str = "topn",
    top_n: int = None,
    final_round: int = 10,
    first_round_strategy: str = "random",
    embedding_type: str = None,
    explicit_variants: Optional[List[str]] = None,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    ucb_beta: float = DEFAULT_UCB_BETA,
    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT,
    redundancy_weight: float = DEFAULT_REDUNDANCY_WEIGHT,
    thompson_scale: float = DEFAULT_THOMPSON_SCALE,
    mixed_exploit_frac: float = DEFAULT_MIXED_EXPLOIT_FRAC,
    mixed_exploit_schedule: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]] = None,
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
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
    success_filter_quantile: float = DEFAULT_SUCCESS_FILTER_QUANTILE,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT,
    success_explore_success_quantile: float = DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE,
    success_explore_pred_quantile: float = DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE,
    safe_max_replace: int = DEFAULT_SAFE_MAX_REPLACE,
    safe_candidate_pool_quantile: float = DEFAULT_SAFE_CANDIDATE_POOL_QUANTILE,
    safe_std_penalty: float = DEFAULT_SAFE_STD_PENALTY,
    safe_novelty_weight: float = DEFAULT_SAFE_NOVELTY_WEIGHT,
    safe_batch_margin: float = DEFAULT_SAFE_BATCH_MARGIN,
    safe_require_batch_score: bool = DEFAULT_SAFE_REQUIRE_BATCH_SCORE,
    pareto_pool_multiplier: float = DEFAULT_PARETO_POOL_MULTIPLIER,
    pareto_lambda_values: Optional[List[float]] = None,
    pareto_balance_rule: str = DEFAULT_PARETO_BALANCE_RULE,
    pareto_min_quality_retention: float = DEFAULT_PARETO_MIN_QUALITY_RETENTION,
    pareto_local_search: bool = DEFAULT_PARETO_LOCAL_SEARCH,
    pareto_local_search_steps: int = DEFAULT_PARETO_LOCAL_SEARCH_STEPS,
    pareto_entropy_base: str = DEFAULT_PARETO_ENTROPY_BASE,
    pareto_force_top1: bool = DEFAULT_PARETO_FORCE_TOP1,
) -> pd.DataFrame:

    output_list = []

    parsed_mixed_schedule_map = _parse_mixed_exploit_schedule(
        mixed_exploit_schedule,
        total_rounds=num_iterations,
    )
    mixed_exploit_schedule_str = _normalize_schedule_spec_string(mixed_exploit_schedule)

    if learning_strategy in {"mixed_ucb_diverse", "mixed_ucb_diverse_schedule", "success_constrained_mixed", "safe_struct_mixed", "guarded_struct_mixed"}:
        if parsed_mixed_schedule_map:
            print(
                f"[mixed schedule] using round-dependent exploit fractions: {parsed_mixed_schedule_map} "
                f"(default fallback={_clamp_mixed_exploit_frac(mixed_exploit_frac):g})"
            )
        else:
            print(
                f"[mixed schedule] no round-specific schedule provided; "
                f"using fixed mixed_exploit_frac={_clamp_mixed_exploit_frac(mixed_exploit_frac):g}"
            )

    for i in range(1, num_simulations + 1):
        iteration_old = None

        num_mutants_per_round_list = []
        first_round_strategy_list = []
        measured_var_list = []
        learning_strategy_list = []
        regression_type_list = []
        embedding_type_list = []
        simulation_list = []
        round_list = []
        test_error_list = []
        train_error_list = []
        train_r_squared_list = []
        test_r_squared_list = []
        alpha_list = []
        median_activity_scaled_list = []
        top_variant_list = []
        top_final_round_variants_list = []
        top_activity_scaled_list = []
        spearman_corr_list = []
        activity_binary_percentage_list = []
        this_round_variants_list = []
        next_round_variants_list = []

        avg_selected_pred_mean_list = []
        avg_selected_pred_std_list = []
        avg_selected_novelty_to_tested_list = []
        avg_selected_ucb_score_base_list = []
        avg_selected_pairwise_distance_list = []
        avg_selected_min_pairwise_distance_list = []
        avg_selected_pred_mean_base_list = []
        avg_selected_head_rerank_score_list = []
        avg_selected_head_rerank_correction_list = []
        frac_selected_head_rerank_applied_list = []

        selected_exploit_count_list = []
        selected_explore_count_list = []
        effective_mixed_exploit_frac_list = []
        mixed_schedule_applied_list = []
        mixed_exploit_schedule_list = []

        ensemble_size_list = []
        bootstrap_sample_frac_list = []
        ucb_beta_list = []
        novelty_weight_list = []
        redundancy_weight_list = []
        mixed_exploit_frac_list = []

        enable_tail_weighting_list = []
        tail_weight_mode_list = []
        tail_weight_quantile_list = []
        tail_weight_strength_list = []
        tail_weight_max_list = []
        enable_head_rerank_list = []
        head_rerank_pool_size_list = []
        head_rerank_weight_list = []
        head_rerank_tail_quantile_list = []
        enable_success_head_list = []
        success_label_mode_list = []
        success_threshold_list = []
        success_positive_quantile_list = []
        success_weight_list = []
        success_filter_quantile_list = []
        success_mode_list = []
        success_filter_applied_list = []
        success_candidates_before_filter_list = []
        success_candidates_after_filter_list = []
        success_gate_applied_list = []
        success_gate_success_cutoff_list = []
        success_gate_pred_cutoff_list = []
        success_head_available_for_selection_list = []
        success_acq_success_weight_list = []
        success_acq_pred_weight_list = []
        success_acq_std_penalty_list = []
        success_acq_novelty_weight_list = []
        success_explore_success_quantile_list = []
        success_explore_pred_quantile_list = []
        safe_guard_applied_list = []
        safe_fallback_to_topn_list = []
        safe_candidate_pool_size_list = []
        safe_replaced_count_list = []
        safe_topn_batch_score_list = []
        safe_candidate_batch_score_list = []
        safe_max_replace_list = []
        safe_candidate_pool_quantile_list = []
        safe_std_penalty_list = []
        safe_novelty_weight_list = []
        safe_batch_margin_list = []
        safe_require_batch_score_list = []
        avg_selected_success_prob_list = []
        min_selected_success_prob_list = []
        avg_selected_success_prob_std_list = []
        avg_selected_success_score_base_list = []
        frac_selected_success_head_enabled_list = []
        selected_position_entropy_list = []
        selected_unique_positions_list = []
        selected_max_position_fraction_list = []
        selected_unparsed_variant_count_list = []

        pareto_applied_list = []
        pareto_fallback_to_topn_list = []
        pareto_pool_size_list = []
        pareto_front_size_list = []
        pareto_batches_evaluated_list = []
        pareto_selected_lambda_list = []
        pareto_selected_quality_list = []
        pareto_selected_entropy_list = []
        pareto_selected_mean_pred_list = []
        pareto_topn_quality_list = []
        pareto_topn_entropy_list = []
        pareto_topn_mean_pred_list = []
        pareto_quality_retention_list = []
        pareto_entropy_gain_list = []
        pareto_selected_unique_positions_list = []
        pareto_selected_max_position_fraction_list = []
        pareto_topn_unique_positions_list = []
        pareto_topn_max_position_fraction_list = []
        pareto_unparsed_variant_count_list = []
        pareto_pool_multiplier_list = []
        pareto_lambda_values_list = []
        pareto_balance_rule_list = []
        pareto_min_quality_retention_list = []
        pareto_local_search_list = []
        pareto_local_search_steps_list = []
        pareto_entropy_base_list = []
        pareto_force_top1_list = []

        j = 0
        labels_new = None
        iteration_new = None

        while j <= num_iterations:
            if j == 0:
                labels_new, iteration_new, this_round_variants = first_round(
                    labels,
                    embeddings,
                    explicit_variants=explicit_variants,
                    num_mutants_per_round=num_mutants_per_round,
                    first_round_strategy=first_round_strategy,
                    embedding_type=embedding_type,
                    random_seed=i,
                )

                simulation_list.append(i)
                round_list.append(j)
                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                measured_var_list.append(measured_var)
                learning_strategy_list.append(learning_strategy)
                regression_type_list.append(regression_type)
                embedding_type_list.append(embedding_type)

                test_error_list.append("None")
                train_error_list.append("None")
                train_r_squared_list.append("None")
                test_r_squared_list.append("None")
                alpha_list.append("None")
                median_activity_scaled_list.append("None")
                top_activity_scaled_list.append("None")
                top_variant_list.append("None")
                top_final_round_variants_list.append("None")
                activity_binary_percentage_list.append("None")
                spearman_corr_list.append("None")
                this_round_variants_list.append("None")
                next_round_variants_list.append(",".join(map(str, this_round_variants)))

                avg_selected_pred_mean_list.append("None")
                avg_selected_pred_std_list.append("None")
                avg_selected_novelty_to_tested_list.append("None")
                avg_selected_ucb_score_base_list.append("None")
                avg_selected_pairwise_distance_list.append("None")
                avg_selected_min_pairwise_distance_list.append("None")
                avg_selected_pred_mean_base_list.append("None")
                avg_selected_head_rerank_score_list.append("None")
                avg_selected_head_rerank_correction_list.append("None")
                frac_selected_head_rerank_applied_list.append("None")
                avg_selected_success_prob_list.append("None")
                min_selected_success_prob_list.append("None")
                avg_selected_success_prob_std_list.append("None")
                avg_selected_success_score_base_list.append("None")
                frac_selected_success_head_enabled_list.append("None")
                selected_position_entropy_list.append("None")
                selected_unique_positions_list.append("None")
                selected_max_position_fraction_list.append("None")
                selected_unparsed_variant_count_list.append("None")

                selected_exploit_count_list.append("None")
                selected_explore_count_list.append("None")
                effective_mixed_exploit_frac_list.append("None")
                mixed_schedule_applied_list.append("None")
                mixed_exploit_schedule_list.append(mixed_exploit_schedule_str if mixed_exploit_schedule_str is not None else "None")

                ensemble_size_list.append(ensemble_size)
                bootstrap_sample_frac_list.append(bootstrap_sample_frac)
                ucb_beta_list.append(ucb_beta)
                novelty_weight_list.append(novelty_weight)
                redundancy_weight_list.append(redundancy_weight)
                mixed_exploit_frac_list.append(mixed_exploit_frac)

                enable_tail_weighting_list.append(enable_tail_weighting)
                tail_weight_mode_list.append(tail_weight_mode)
                tail_weight_quantile_list.append(tail_weight_quantile)
                tail_weight_strength_list.append(tail_weight_strength)
                tail_weight_max_list.append(tail_weight_max)
                enable_head_rerank_list.append(enable_head_rerank)
                head_rerank_pool_size_list.append(head_rerank_pool_size)
                head_rerank_weight_list.append(head_rerank_weight)
                head_rerank_tail_quantile_list.append(head_rerank_tail_quantile)
                enable_success_head_list.append(enable_success_head)
                success_label_mode_list.append(success_label_mode)
                success_threshold_list.append(success_threshold)
                success_positive_quantile_list.append(success_positive_quantile)
                success_weight_list.append(success_weight)
                success_filter_quantile_list.append(success_filter_quantile)
                success_mode_list.append(success_mode)
                success_filter_applied_list.append("None")
                success_candidates_before_filter_list.append("None")
                success_candidates_after_filter_list.append("None")
                success_gate_applied_list.append("None")
                success_gate_success_cutoff_list.append("None")
                success_gate_pred_cutoff_list.append("None")
                success_head_available_for_selection_list.append("None")
                success_acq_success_weight_list.append(success_acq_success_weight)
                success_acq_pred_weight_list.append(success_acq_pred_weight)
                success_acq_std_penalty_list.append(success_acq_std_penalty)
                success_acq_novelty_weight_list.append(success_acq_novelty_weight)
                success_explore_success_quantile_list.append(success_explore_success_quantile)
                success_explore_pred_quantile_list.append(success_explore_pred_quantile)
                safe_guard_applied_list.append("None")
                safe_fallback_to_topn_list.append("None")
                safe_candidate_pool_size_list.append("None")
                safe_replaced_count_list.append("None")
                safe_topn_batch_score_list.append("None")
                safe_candidate_batch_score_list.append("None")
                safe_max_replace_list.append(safe_max_replace)
                safe_candidate_pool_quantile_list.append(safe_candidate_pool_quantile)
                safe_std_penalty_list.append(safe_std_penalty)
                safe_novelty_weight_list.append(safe_novelty_weight)
                safe_batch_margin_list.append(safe_batch_margin)
                safe_require_batch_score_list.append(safe_require_batch_score)
                pareto_applied_list.append("None")
                pareto_fallback_to_topn_list.append("None")
                pareto_pool_size_list.append("None")
                pareto_front_size_list.append("None")
                pareto_batches_evaluated_list.append("None")
                pareto_selected_lambda_list.append("None")
                pareto_selected_quality_list.append("None")
                pareto_selected_entropy_list.append("None")
                pareto_selected_mean_pred_list.append("None")
                pareto_topn_quality_list.append("None")
                pareto_topn_entropy_list.append("None")
                pareto_topn_mean_pred_list.append("None")
                pareto_quality_retention_list.append("None")
                pareto_entropy_gain_list.append("None")
                pareto_selected_unique_positions_list.append("None")
                pareto_selected_max_position_fraction_list.append("None")
                pareto_topn_unique_positions_list.append("None")
                pareto_topn_max_position_fraction_list.append("None")
                pareto_unparsed_variant_count_list.append("None")
                pareto_pool_multiplier_list.append(pareto_pool_multiplier)
                pareto_lambda_values_list.append(",".join(map(str, pareto_lambda_values or DEFAULT_PARETO_LAMBDA_VALUES)))
                pareto_balance_rule_list.append(pareto_balance_rule)
                pareto_min_quality_retention_list.append(pareto_min_quality_retention)
                pareto_local_search_list.append(pareto_local_search)
                pareto_local_search_steps_list.append(pareto_local_search_steps)
                pareto_entropy_base_list.append(pareto_entropy_base)
                pareto_force_top1_list.append(pareto_force_top1)

                j += 1
                continue

            iteration_old = iteration_new

            (
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
                df_test_new,
                this_round_variants,
            ) = top_layer(
                iter_train=iteration_old["iteration"].unique().tolist(),
                iter_test=None,
                embeddings_pd=embeddings,
                labels_pd=labels_new,
                measured_var=measured_var,
                regression_type=regression_type,
                top_n=top_n,
                final_round=final_round,
                ensemble_size=ensemble_size,
                bootstrap_sample_frac=bootstrap_sample_frac,
                ucb_beta=ucb_beta,
                random_seed=(i * 1000 + j),
                enable_tail_weighting=enable_tail_weighting,
                tail_weight_mode=tail_weight_mode,
                tail_weight_quantile=tail_weight_quantile,
                tail_weight_strength=tail_weight_strength,
                tail_weight_max=tail_weight_max,
                enable_head_rerank=enable_head_rerank,
                head_rerank_pool_size=head_rerank_pool_size,
                head_rerank_weight=head_rerank_weight,
                head_rerank_tail_quantile=head_rerank_tail_quantile,
                enable_success_head=enable_success_head,
                success_label_mode=success_label_mode,
                success_threshold=success_threshold,
                success_positive_quantile=success_positive_quantile,
            )

            iteration_new_ids, selection_meta = _select_next_round_variants(
                df_test_new=df_test_new,
                learning_strategy=learning_strategy,
                num_mutants_per_round=num_mutants_per_round,
                random_seed=(i * 1000 + j),
                embeddings=embeddings,
                ucb_beta=ucb_beta,
                novelty_weight=novelty_weight,
                redundancy_weight=redundancy_weight,
                thompson_scale=thompson_scale,
                mixed_exploit_frac=mixed_exploit_frac,
                current_round=j,
                mixed_exploit_schedule=mixed_exploit_schedule,
                parsed_mixed_schedule_map=parsed_mixed_schedule_map,
                success_weight=success_weight,
                success_filter_quantile=success_filter_quantile,
                success_mode=success_mode,
                success_acq_success_weight=success_acq_success_weight,
                success_acq_pred_weight=success_acq_pred_weight,
                success_acq_std_penalty=success_acq_std_penalty,
                success_acq_novelty_weight=success_acq_novelty_weight,
                success_explore_success_quantile=success_explore_success_quantile,
                success_explore_pred_quantile=success_explore_pred_quantile,
                safe_max_replace=safe_max_replace,
                safe_candidate_pool_quantile=safe_candidate_pool_quantile,
                safe_std_penalty=safe_std_penalty,
                safe_novelty_weight=safe_novelty_weight,
                safe_batch_margin=safe_batch_margin,
                safe_require_batch_score=safe_require_batch_score,
                pareto_pool_multiplier=pareto_pool_multiplier,
                pareto_lambda_values=pareto_lambda_values,
                pareto_balance_rule=pareto_balance_rule,
                pareto_min_quality_retention=pareto_min_quality_retention,
                pareto_local_search=pareto_local_search,
                pareto_local_search_steps=pareto_local_search_steps,
                pareto_entropy_base=pareto_entropy_base,
                pareto_force_top1=pareto_force_top1,
            )

            batch_stats = _summarize_selected_batch(
                df_test_new=df_test_new,
                selected_variants=iteration_new_ids,
                embeddings=embeddings,
            )

            iteration_new = pd.DataFrame(
                {"variant": iteration_new_ids, "iteration": j}
            )
            iteration_new = pd.concat([iteration_new, iteration_old], ignore_index=True)
            labels_new = pd.merge(labels, iteration_new, on="variant", how="left")

            simulation_list.append(i)
            round_list.append(j)
            num_mutants_per_round_list.append(num_mutants_per_round)
            first_round_strategy_list.append(first_round_strategy)
            measured_var_list.append(measured_var)
            learning_strategy_list.append(learning_strategy)
            regression_type_list.append(regression_type)
            embedding_type_list.append(embedding_type)
            test_error_list.append(test_error)
            train_error_list.append(train_error)
            train_r_squared_list.append(train_r_squared)
            test_r_squared_list.append(test_r_squared)
            alpha_list.append(alpha)
            median_activity_scaled_list.append(median_activity_scaled)
            top_activity_scaled_list.append(top_activity_scaled)
            top_variant_list.append(top_variant)
            top_final_round_variants_list.append(top_final_round_variants)
            activity_binary_percentage_list.append(activity_binary_percentage)
            spearman_corr_list.append(spearman_corr)
            this_round_variants_list.append(",".join(map(str, iteration_old["variant"].tolist())))
            next_round_variants_list.append(",".join(map(str, iteration_new_ids)))

            avg_selected_pred_mean_list.append(batch_stats["avg_selected_pred_mean"])
            avg_selected_pred_std_list.append(batch_stats["avg_selected_pred_std"])
            avg_selected_novelty_to_tested_list.append(batch_stats["avg_selected_novelty_to_tested"])
            avg_selected_ucb_score_base_list.append(batch_stats["avg_selected_ucb_score_base"])
            avg_selected_pairwise_distance_list.append(batch_stats["avg_selected_pairwise_distance"])
            avg_selected_min_pairwise_distance_list.append(batch_stats["avg_selected_min_pairwise_distance"])
            avg_selected_pred_mean_base_list.append(batch_stats["avg_selected_pred_mean_base"])
            avg_selected_head_rerank_score_list.append(batch_stats["avg_selected_head_rerank_score"])
            avg_selected_head_rerank_correction_list.append(batch_stats["avg_selected_head_rerank_correction"])
            frac_selected_head_rerank_applied_list.append(batch_stats["frac_selected_head_rerank_applied"])
            avg_selected_success_prob_list.append(batch_stats["avg_selected_success_prob"])
            min_selected_success_prob_list.append(batch_stats["min_selected_success_prob"])
            avg_selected_success_prob_std_list.append(batch_stats["avg_selected_success_prob_std"])
            avg_selected_success_score_base_list.append(batch_stats["avg_selected_success_score_base"])
            frac_selected_success_head_enabled_list.append(batch_stats["frac_selected_success_head_enabled"])
            selected_position_entropy_list.append(batch_stats["selected_position_entropy"])
            selected_unique_positions_list.append(batch_stats["selected_unique_positions"])
            selected_max_position_fraction_list.append(batch_stats["selected_max_position_fraction"])
            selected_unparsed_variant_count_list.append(batch_stats["selected_unparsed_variant_count"])

            selected_exploit_count_list.append(selection_meta.get("selected_exploit_count", np.nan))
            selected_explore_count_list.append(selection_meta.get("selected_explore_count", np.nan))
            effective_mixed_exploit_frac_list.append(selection_meta.get("effective_mixed_exploit_frac", np.nan))
            mixed_schedule_applied_list.append(selection_meta.get("mixed_schedule_applied", False))
            mixed_exploit_schedule_list.append(mixed_exploit_schedule_str if mixed_exploit_schedule_str is not None else "")
            success_filter_applied_list.append(selection_meta.get("success_filter_applied", False))
            success_candidates_before_filter_list.append(selection_meta.get("success_candidates_before_filter", np.nan))
            success_candidates_after_filter_list.append(selection_meta.get("success_candidates_after_filter", np.nan))
            success_gate_applied_list.append(selection_meta.get("success_gate_applied", False))
            success_gate_success_cutoff_list.append(selection_meta.get("success_gate_success_cutoff", np.nan))
            success_gate_pred_cutoff_list.append(selection_meta.get("success_gate_pred_cutoff", np.nan))
            success_head_available_for_selection_list.append(selection_meta.get("success_head_available_for_selection", np.nan))
            success_acq_success_weight_list.append(selection_meta.get("success_acq_success_weight", success_acq_success_weight))
            success_acq_pred_weight_list.append(selection_meta.get("success_acq_pred_weight", success_acq_pred_weight))
            success_acq_std_penalty_list.append(selection_meta.get("success_acq_std_penalty", success_acq_std_penalty))
            success_acq_novelty_weight_list.append(selection_meta.get("success_acq_novelty_weight", success_acq_novelty_weight))
            success_explore_success_quantile_list.append(selection_meta.get("success_explore_success_quantile", success_explore_success_quantile))
            success_explore_pred_quantile_list.append(selection_meta.get("success_explore_pred_quantile", success_explore_pred_quantile))
            safe_guard_applied_list.append(selection_meta.get("safe_guard_applied", False))
            safe_fallback_to_topn_list.append(selection_meta.get("safe_fallback_to_topn", False))
            safe_candidate_pool_size_list.append(selection_meta.get("safe_candidate_pool_size", np.nan))
            safe_replaced_count_list.append(selection_meta.get("safe_replaced_count", np.nan))
            safe_topn_batch_score_list.append(selection_meta.get("safe_topn_batch_score", np.nan))
            safe_candidate_batch_score_list.append(selection_meta.get("safe_candidate_batch_score", np.nan))
            safe_max_replace_list.append(selection_meta.get("safe_max_replace", safe_max_replace))
            safe_candidate_pool_quantile_list.append(selection_meta.get("safe_candidate_pool_quantile", safe_candidate_pool_quantile))
            safe_std_penalty_list.append(selection_meta.get("safe_std_penalty", safe_std_penalty))
            safe_novelty_weight_list.append(selection_meta.get("safe_novelty_weight", safe_novelty_weight))
            safe_batch_margin_list.append(selection_meta.get("safe_batch_margin", safe_batch_margin))
            safe_require_batch_score_list.append(selection_meta.get("safe_require_batch_score", safe_require_batch_score))
            pareto_applied_list.append(selection_meta.get("pareto_applied", False))
            pareto_fallback_to_topn_list.append(selection_meta.get("pareto_fallback_to_topn", False))
            pareto_pool_size_list.append(selection_meta.get("pareto_pool_size", np.nan))
            pareto_front_size_list.append(selection_meta.get("pareto_front_size", np.nan))
            pareto_batches_evaluated_list.append(selection_meta.get("pareto_batches_evaluated", np.nan))
            pareto_selected_lambda_list.append(selection_meta.get("pareto_selected_lambda", np.nan))
            pareto_selected_quality_list.append(selection_meta.get("pareto_selected_quality", np.nan))
            pareto_selected_entropy_list.append(selection_meta.get("pareto_selected_entropy", np.nan))
            pareto_selected_mean_pred_list.append(selection_meta.get("pareto_selected_mean_pred", np.nan))
            pareto_topn_quality_list.append(selection_meta.get("pareto_topn_quality", np.nan))
            pareto_topn_entropy_list.append(selection_meta.get("pareto_topn_entropy", np.nan))
            pareto_topn_mean_pred_list.append(selection_meta.get("pareto_topn_mean_pred", np.nan))
            pareto_quality_retention_list.append(selection_meta.get("pareto_quality_retention", np.nan))
            pareto_entropy_gain_list.append(selection_meta.get("pareto_entropy_gain", np.nan))
            pareto_selected_unique_positions_list.append(selection_meta.get("pareto_selected_unique_positions", np.nan))
            pareto_selected_max_position_fraction_list.append(selection_meta.get("pareto_selected_max_position_fraction", np.nan))
            pareto_topn_unique_positions_list.append(selection_meta.get("pareto_topn_unique_positions", np.nan))
            pareto_topn_max_position_fraction_list.append(selection_meta.get("pareto_topn_max_position_fraction", np.nan))
            pareto_unparsed_variant_count_list.append(selection_meta.get("pareto_unparsed_variant_count", np.nan))
            pareto_pool_multiplier_list.append(selection_meta.get("pareto_pool_multiplier", pareto_pool_multiplier))
            pareto_lambda_values_list.append(",".join(map(str, pareto_lambda_values or DEFAULT_PARETO_LAMBDA_VALUES)))
            pareto_balance_rule_list.append(selection_meta.get("pareto_balance_rule", pareto_balance_rule))
            pareto_min_quality_retention_list.append(selection_meta.get("pareto_min_quality_retention", pareto_min_quality_retention))
            pareto_local_search_list.append(selection_meta.get("pareto_local_search", pareto_local_search))
            pareto_local_search_steps_list.append(selection_meta.get("pareto_local_search_steps", pareto_local_search_steps))
            pareto_entropy_base_list.append(selection_meta.get("pareto_entropy_base", pareto_entropy_base))
            pareto_force_top1_list.append(selection_meta.get("pareto_force_top1", pareto_force_top1))

            ensemble_size_list.append(ensemble_size)
            bootstrap_sample_frac_list.append(bootstrap_sample_frac)
            ucb_beta_list.append(ucb_beta)
            novelty_weight_list.append(novelty_weight)
            redundancy_weight_list.append(redundancy_weight)
            mixed_exploit_frac_list.append(mixed_exploit_frac)

            enable_tail_weighting_list.append(enable_tail_weighting)
            tail_weight_mode_list.append(tail_weight_mode)
            tail_weight_quantile_list.append(tail_weight_quantile)
            tail_weight_strength_list.append(tail_weight_strength)
            tail_weight_max_list.append(tail_weight_max)
            enable_head_rerank_list.append(enable_head_rerank)
            head_rerank_pool_size_list.append(head_rerank_pool_size)
            head_rerank_weight_list.append(head_rerank_weight)
            head_rerank_tail_quantile_list.append(head_rerank_tail_quantile)
            enable_success_head_list.append(enable_success_head)
            success_label_mode_list.append(success_label_mode)
            success_threshold_list.append(success_threshold)
            success_positive_quantile_list.append(success_positive_quantile)
            success_weight_list.append(success_weight)
            success_filter_quantile_list.append(success_filter_quantile)
            success_mode_list.append(success_mode)

            j += 1

        df_metrics = pd.DataFrame(
            {
                "simulation_num": simulation_list,
                "round_num": round_list,
                "num_mutants_per_round": num_mutants_per_round_list,
                "first_round_strategy": first_round_strategy_list,
                "measured_var": measured_var_list,
                "learning_strategy": learning_strategy_list,
                "regression_type": regression_type_list,
                "embedding_type": embedding_type_list,
                "test_error": test_error_list,
                "train_error": train_error_list,
                "train_r_squared": train_r_squared_list,
                "test_r_squared": test_r_squared_list,
                "alpha": alpha_list,
                "spearman_corr": spearman_corr_list,
                "median_activity_scaled": median_activity_scaled_list,
                "top_activity_scaled": top_activity_scaled_list,
                "activity_binary_percentage": activity_binary_percentage_list,
                "top_variant": top_variant_list,
                "top_final_round_variants": top_final_round_variants_list,
                "this_round_variants": this_round_variants_list,
                "next_round_variants": next_round_variants_list,

                "avg_selected_pred_mean": avg_selected_pred_mean_list,
                "avg_selected_pred_mean_base": avg_selected_pred_mean_base_list,
                "avg_selected_pred_std": avg_selected_pred_std_list,
                "avg_selected_novelty_to_tested": avg_selected_novelty_to_tested_list,
                "avg_selected_ucb_score_base": avg_selected_ucb_score_base_list,
                "avg_selected_pairwise_distance": avg_selected_pairwise_distance_list,
                "avg_selected_min_pairwise_distance": avg_selected_min_pairwise_distance_list,
                "avg_selected_head_rerank_score": avg_selected_head_rerank_score_list,
                "avg_selected_head_rerank_correction": avg_selected_head_rerank_correction_list,
                "frac_selected_head_rerank_applied": frac_selected_head_rerank_applied_list,
                "avg_selected_success_prob": avg_selected_success_prob_list,
                "min_selected_success_prob": min_selected_success_prob_list,
                "avg_selected_success_prob_std": avg_selected_success_prob_std_list,
                "avg_selected_success_score_base": avg_selected_success_score_base_list,
                "frac_selected_success_head_enabled": frac_selected_success_head_enabled_list,
                "selected_position_entropy": selected_position_entropy_list,
                "selected_unique_positions": selected_unique_positions_list,
                "selected_max_position_fraction": selected_max_position_fraction_list,
                "selected_unparsed_variant_count": selected_unparsed_variant_count_list,

                "selected_exploit_count": selected_exploit_count_list,
                "selected_explore_count": selected_explore_count_list,
                "effective_mixed_exploit_frac": effective_mixed_exploit_frac_list,
                "mixed_schedule_applied": mixed_schedule_applied_list,
                "mixed_exploit_schedule": mixed_exploit_schedule_list,
                "success_filter_applied": success_filter_applied_list,
                "success_candidates_before_filter": success_candidates_before_filter_list,
                "success_candidates_after_filter": success_candidates_after_filter_list,
                "success_gate_applied": success_gate_applied_list,
                "success_gate_success_cutoff": success_gate_success_cutoff_list,
                "success_gate_pred_cutoff": success_gate_pred_cutoff_list,
                "success_head_available_for_selection": success_head_available_for_selection_list,

                "ensemble_size": ensemble_size_list,
                "bootstrap_sample_frac": bootstrap_sample_frac_list,
                "ucb_beta": ucb_beta_list,
                "novelty_weight": novelty_weight_list,
                "redundancy_weight": redundancy_weight_list,
                "mixed_exploit_frac": mixed_exploit_frac_list,
                "enable_tail_weighting": enable_tail_weighting_list,
                "tail_weight_mode": tail_weight_mode_list,
                "tail_weight_quantile": tail_weight_quantile_list,
                "tail_weight_strength": tail_weight_strength_list,
                "tail_weight_max": tail_weight_max_list,
                "enable_head_rerank": enable_head_rerank_list,
                "head_rerank_pool_size": head_rerank_pool_size_list,
                "head_rerank_weight": head_rerank_weight_list,
                "head_rerank_tail_quantile": head_rerank_tail_quantile_list,
                "enable_success_head": enable_success_head_list,
                "success_label_mode": success_label_mode_list,
                "success_threshold": success_threshold_list,
                "success_positive_quantile": success_positive_quantile_list,
                "success_weight": success_weight_list,
                "success_filter_quantile": success_filter_quantile_list,
                "success_mode": success_mode_list,
                "success_acq_success_weight": success_acq_success_weight_list,
                "success_acq_pred_weight": success_acq_pred_weight_list,
                "success_acq_std_penalty": success_acq_std_penalty_list,
                "success_acq_novelty_weight": success_acq_novelty_weight_list,
                "success_explore_success_quantile": success_explore_success_quantile_list,
                "success_explore_pred_quantile": success_explore_pred_quantile_list,
                "safe_guard_applied": safe_guard_applied_list,
                "safe_fallback_to_topn": safe_fallback_to_topn_list,
                "safe_candidate_pool_size": safe_candidate_pool_size_list,
                "safe_replaced_count": safe_replaced_count_list,
                "safe_topn_batch_score": safe_topn_batch_score_list,
                "safe_candidate_batch_score": safe_candidate_batch_score_list,
                "safe_max_replace": safe_max_replace_list,
                "safe_candidate_pool_quantile": safe_candidate_pool_quantile_list,
                "safe_std_penalty": safe_std_penalty_list,
                "safe_novelty_weight": safe_novelty_weight_list,
                "safe_batch_margin": safe_batch_margin_list,
                "safe_require_batch_score": safe_require_batch_score_list,
                "pareto_applied": pareto_applied_list,
                "pareto_fallback_to_topn": pareto_fallback_to_topn_list,
                "pareto_pool_size": pareto_pool_size_list,
                "pareto_front_size": pareto_front_size_list,
                "pareto_batches_evaluated": pareto_batches_evaluated_list,
                "pareto_selected_lambda": pareto_selected_lambda_list,
                "pareto_selected_quality": pareto_selected_quality_list,
                "pareto_selected_entropy": pareto_selected_entropy_list,
                "pareto_selected_mean_pred": pareto_selected_mean_pred_list,
                "pareto_topn_quality": pareto_topn_quality_list,
                "pareto_topn_entropy": pareto_topn_entropy_list,
                "pareto_topn_mean_pred": pareto_topn_mean_pred_list,
                "pareto_quality_retention": pareto_quality_retention_list,
                "pareto_entropy_gain": pareto_entropy_gain_list,
                "pareto_selected_unique_positions": pareto_selected_unique_positions_list,
                "pareto_selected_max_position_fraction": pareto_selected_max_position_fraction_list,
                "pareto_topn_unique_positions": pareto_topn_unique_positions_list,
                "pareto_topn_max_position_fraction": pareto_topn_max_position_fraction_list,
                "pareto_unparsed_variant_count": pareto_unparsed_variant_count_list,
                "pareto_pool_multiplier": pareto_pool_multiplier_list,
                "pareto_lambda_values": pareto_lambda_values_list,
                "pareto_balance_rule": pareto_balance_rule_list,
                "pareto_min_quality_retention": pareto_min_quality_retention_list,
                "pareto_local_search": pareto_local_search_list,
                "pareto_local_search_steps": pareto_local_search_steps_list,
                "pareto_entropy_base": pareto_entropy_base_list,
                "pareto_force_top1": pareto_force_top1_list,
            }
        )
        output_list.append(df_metrics)

    return pd.concat(output_list, ignore_index=True)

def grid_search(
    dataset_name: str,
    experiment_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    num_simulations: int,
    num_iterations: List[int],
    measured_var: List[str],
    learning_strategies: List[str],
    num_mutants_per_round: List[int],
    num_final_round_mutants: int,
    first_round_strategies: List[str],
    embedding_types: List[str],
    pca_components: List[int],
    regression_types: List[str],
    embeddings_file_type: str,
    output_dir: str,
    embeddings_type_pt: Optional[str] = None,
    struct_path: Optional[str] = None,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    ucb_beta: float = DEFAULT_UCB_BETA,
    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT,
    redundancy_weight: float = DEFAULT_REDUNDANCY_WEIGHT,
    thompson_scale: float = DEFAULT_THOMPSON_SCALE,
    mixed_exploit_frac: float = DEFAULT_MIXED_EXPLOIT_FRAC,
    mixed_exploit_schedule: Optional[Union[str, Dict[int, float], List[float], Tuple[float, ...]]] = None,
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
    success_weight: float = DEFAULT_SUCCESS_WEIGHT,
    success_filter_quantile: float = DEFAULT_SUCCESS_FILTER_QUANTILE,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    success_acq_success_weight: float = DEFAULT_SUCCESS_ACQ_SUCCESS_WEIGHT,
    success_acq_pred_weight: float = DEFAULT_SUCCESS_ACQ_PRED_WEIGHT,
    success_acq_std_penalty: float = DEFAULT_SUCCESS_ACQ_STD_PENALTY,
    success_acq_novelty_weight: float = DEFAULT_SUCCESS_ACQ_NOVELTY_WEIGHT,
    success_explore_success_quantile: float = DEFAULT_SUCCESS_EXPLORE_SUCCESS_QUANTILE,
    success_explore_pred_quantile: float = DEFAULT_SUCCESS_EXPLORE_PRED_QUANTILE,
    safe_max_replace: int = DEFAULT_SAFE_MAX_REPLACE,
    safe_candidate_pool_quantile: float = DEFAULT_SAFE_CANDIDATE_POOL_QUANTILE,
    safe_std_penalty: float = DEFAULT_SAFE_STD_PENALTY,
    safe_novelty_weight: float = DEFAULT_SAFE_NOVELTY_WEIGHT,
    safe_batch_margin: float = DEFAULT_SAFE_BATCH_MARGIN,
    safe_require_batch_score: bool = DEFAULT_SAFE_REQUIRE_BATCH_SCORE,
    pareto_pool_multiplier: float = DEFAULT_PARETO_POOL_MULTIPLIER,
    pareto_lambda_values: Optional[List[float]] = None,
    pareto_balance_rule: str = DEFAULT_PARETO_BALANCE_RULE,
    pareto_min_quality_retention: float = DEFAULT_PARETO_MIN_QUALITY_RETENTION,
    pareto_local_search: bool = DEFAULT_PARETO_LOCAL_SEARCH,
    pareto_local_search_steps: int = DEFAULT_PARETO_LOCAL_SEARCH_STEPS,
    pareto_entropy_base: str = DEFAULT_PARETO_ENTROPY_BASE,
    pareto_force_top1: bool = DEFAULT_PARETO_FORCE_TOP1,
) -> None:
    plm_embeddings, struct_df, labels = load_dms_data(
        dataset_name=dataset_name,
        model_name=model_name,
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        embeddings_file_type=embeddings_file_type,
        embeddings_type_pt=embeddings_type_pt,
        struct_path=struct_path,
    )

    if plm_embeddings is None or labels is None:
        print("Data loading failed. Exiting.")
        return

    embeddings_list = _build_embedding_pool(
        plm_embeddings=plm_embeddings,
        struct_df=struct_df,
        embedding_types=embedding_types,
    )

    runnable_embedding_types = [
        etype for etype in embedding_types if etype in embeddings_list
    ]

    if not runnable_embedding_types:
        print("No valid embedding types were generated. Exiting.")
        return

    total_combinations = (
        len(learning_strategies)
        * len(measured_var)
        * len(num_iterations)
        * len(num_mutants_per_round)
        * len(runnable_embedding_types)
        * len(regression_types)
        * len(first_round_strategies)
    )
    print(f"Total runnable combinations: {total_combinations}")

    output_list = []
    combination_count = 0
    start_time = time.time()

    for strategy in learning_strategies:
        for var in measured_var:
            for iterations in num_iterations:
                for mutants_per_round in num_mutants_per_round:
                    for embedding_type in runnable_embedding_types:
                        for regression_type in regression_types:
                            for first_round_strategy in first_round_strategies:
                                combination_count += 1

                                print(
                                    f"[{combination_count}/{total_combinations}] "
                                    f"dataset={dataset_name}, "
                                    f"embedding={embedding_type}, "
                                    f"regression={regression_type}, "
                                    f"strategy={strategy}, "
                                    f"first_round={first_round_strategy}, "
                                    f"iterations={iterations}, "
                                    f"mutants_per_round={mutants_per_round}, "
                                    f"ensemble_size={ensemble_size}, "
                                    f"ucb_beta={ucb_beta}, "
                                    f"novelty_weight={novelty_weight}, "
                                    f"redundancy_weight={redundancy_weight}, "
                                    f"mixed_exploit_frac={mixed_exploit_frac}, "
                                    f"mixed_exploit_schedule={_normalize_schedule_spec_string(mixed_exploit_schedule)}, "
                                    f"enable_tail_weighting={enable_tail_weighting}, "
                                    f"tail_weight_mode={tail_weight_mode}, "
                                    f"tail_weight_quantile={tail_weight_quantile}, "
                                    f"tail_weight_strength={tail_weight_strength}, "
                                    f"tail_weight_max={tail_weight_max}, "
                                    f"enable_head_rerank={enable_head_rerank}, "
                                    f"head_rerank_pool_size={head_rerank_pool_size}, "
                                    f"head_rerank_weight={head_rerank_weight}, "
                                    f"head_rerank_tail_quantile={head_rerank_tail_quantile}, "
                                    f"enable_success_head={enable_success_head}, "
                                    f"success_label_mode={success_label_mode}, "
                                    f"success_threshold={success_threshold}, "
                                    f"success_positive_quantile={success_positive_quantile}, "
                                    f"success_weight={success_weight}, "
                                    f"success_filter_quantile={success_filter_quantile}, "
                                    f"success_mode={success_mode}, "
                                    f"success_acq_success_weight={success_acq_success_weight}, "
                                    f"success_acq_pred_weight={success_acq_pred_weight}, "
                                    f"success_acq_std_penalty={success_acq_std_penalty}, "
                                    f"success_acq_novelty_weight={success_acq_novelty_weight}, "
                                    f"success_explore_success_quantile={success_explore_success_quantile}, "
                                    f"success_explore_pred_quantile={success_explore_pred_quantile}, "
                                    f"safe_max_replace={safe_max_replace}, "
                                    f"safe_candidate_pool_quantile={safe_candidate_pool_quantile}, "
                                    f"safe_std_penalty={safe_std_penalty}, "
                                    f"safe_novelty_weight={safe_novelty_weight}, "
                                    f"safe_batch_margin={safe_batch_margin}, "
                                    f"safe_require_batch_score={safe_require_batch_score}, "
                                    f"pareto_pool_multiplier={pareto_pool_multiplier}, "
                                    f"pareto_lambda_values={pareto_lambda_values or list(DEFAULT_PARETO_LAMBDA_VALUES)}, "
                                    f"pareto_balance_rule={pareto_balance_rule}, "
                                    f"pareto_min_quality_retention={pareto_min_quality_retention}, "
                                    f"pareto_local_search={pareto_local_search}, "
                                    f"pareto_local_search_steps={pareto_local_search_steps}, "
                                    f"pareto_entropy_base={pareto_entropy_base}, "
                                    f"pareto_force_top1={pareto_force_top1}"
                                )

                                output_table = directed_evolution_simulation(
                                    labels=labels,
                                    embeddings=embeddings_list[embedding_type],
                                    num_simulations=num_simulations,
                                    num_iterations=iterations,
                                    num_mutants_per_round=mutants_per_round,
                                    measured_var=var,
                                    regression_type=regression_type,
                                    learning_strategy=strategy,
                                    final_round=num_final_round_mutants,
                                    first_round_strategy=first_round_strategy,
                                    embedding_type=embedding_type,
                                    ensemble_size=ensemble_size,
                                    bootstrap_sample_frac=bootstrap_sample_frac,
                                    ucb_beta=ucb_beta,
                                    novelty_weight=novelty_weight,
                                    redundancy_weight=redundancy_weight,
                                    thompson_scale=thompson_scale,
                                    mixed_exploit_frac=mixed_exploit_frac,
                                    mixed_exploit_schedule=mixed_exploit_schedule,
                                    enable_tail_weighting=enable_tail_weighting,
                                    tail_weight_mode=tail_weight_mode,
                                    tail_weight_quantile=tail_weight_quantile,
                                    tail_weight_strength=tail_weight_strength,
                                    tail_weight_max=tail_weight_max,
                                    enable_head_rerank=enable_head_rerank,
                                    head_rerank_pool_size=head_rerank_pool_size,
                                    head_rerank_weight=head_rerank_weight,
                                    head_rerank_tail_quantile=head_rerank_tail_quantile,
                                    enable_success_head=enable_success_head,
                                    success_label_mode=success_label_mode,
                                    success_threshold=success_threshold,
                                    success_positive_quantile=success_positive_quantile,
                                    success_weight=success_weight,
                                    success_filter_quantile=success_filter_quantile,
                                    success_mode=success_mode,
                                    success_acq_success_weight=success_acq_success_weight,
                                    success_acq_pred_weight=success_acq_pred_weight,
                                    success_acq_std_penalty=success_acq_std_penalty,
                                    success_acq_novelty_weight=success_acq_novelty_weight,
                                    success_explore_success_quantile=success_explore_success_quantile,
                                    success_explore_pred_quantile=success_explore_pred_quantile,
                                    safe_max_replace=safe_max_replace,
                                    safe_candidate_pool_quantile=safe_candidate_pool_quantile,
                                    safe_std_penalty=safe_std_penalty,
                                    safe_novelty_weight=safe_novelty_weight,
                                    safe_batch_margin=safe_batch_margin,
                                    safe_require_batch_score=safe_require_batch_score,
                                    pareto_pool_multiplier=pareto_pool_multiplier,
                                    pareto_lambda_values=pareto_lambda_values,
                                    pareto_balance_rule=pareto_balance_rule,
                                    pareto_min_quality_retention=pareto_min_quality_retention,
                                    pareto_local_search=pareto_local_search,
                                    pareto_local_search_steps=pareto_local_search_steps,
                                    pareto_entropy_base=pareto_entropy_base,
                                    pareto_force_top1=pareto_force_top1,
                                )

                                output_list.append(output_table)

    if not output_list:
        print("No simulation results generated. Check parameter inputs.")
        return

    df_results = pd.concat(output_list, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{embeddings_type_pt}" if embeddings_type_pt else ""
    save_path = f"{output_dir}/{dataset_name}_{model_name}_{experiment_name}{suffix}.csv"
    df_results.to_csv(save_path, index=False)

    elapsed = time.time() - start_time
    print(f"Execution complete in {elapsed:.2f}s. Saved to {save_path}")


def evolve_experimental(
    protein_name: str,
    round_name: str,
    embeddings_base_path: str,
    embeddings_file_name: str,
    round_base_path: str,
    round_file_names: List[str],
    wt_fasta_path: str,
    rename_WT: bool = False,
    number_of_variants: int = 12,
    output_dir: str = "/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/exp_results/",
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    ucb_beta: float = DEFAULT_UCB_BETA,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Processing {protein_name} - {round_name}")

    embeddings = load_experimental_embeddings(
        embeddings_base_path, embeddings_file_name, rename_WT
    )
    embeddings = _prepare_plm_block(embeddings)
    print(f"Embeddings loaded: {embeddings.shape}")

    all_experimental_data = []
    for round_file_name in round_file_names:
        experimental_data = load_experimental_data(
            round_base_path, round_file_name, wt_fasta_path
        )
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")

    iteration, labels = create_iteration_dataframes(
        all_experimental_data, embeddings.index
    )
    print(f"iteration shape: {iteration.shape}")
    print(f"Labels shape: {labels.shape}")

    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration["iteration"].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var="activity",
        regression_type="randomforest",
        experimental=True,
        ensemble_size=ensemble_size,
        bootstrap_sample_frac=bootstrap_sample_frac,
        ucb_beta=ucb_beta,
        enable_tail_weighting=enable_tail_weighting,
        tail_weight_mode=tail_weight_mode,
        tail_weight_quantile=tail_weight_quantile,
        tail_weight_strength=tail_weight_strength,
        tail_weight_max=tail_weight_max,
        enable_head_rerank=enable_head_rerank,
        head_rerank_pool_size=head_rerank_pool_size,
        head_rerank_weight=head_rerank_weight,
        head_rerank_tail_quantile=head_rerank_tail_quantile,
        enable_success_head=enable_success_head,
        success_label_mode=success_label_mode,
        success_threshold=success_threshold,
        success_positive_quantile=success_positive_quantile,
    )

    pred_col = "pred_mean" if "pred_mean" in df_test.columns else "y_pred"

    print(f"\nTested variants in this round: {len(this_round_variants)}")
    print(this_round_variants)
    print(f"\nTop {number_of_variants} variants predicted by the model:")
    print(df_test.sort_values(by=[pred_col], ascending=False).head(number_of_variants))

    if output_dir is not None:
        output_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, "iteration.csv"))
        this_round_variants.to_csv(os.path.join(output_dir, "this_round_variants.csv"))
        df_test = df_test.sort_values(by=[pred_col], ascending=False)
        df_test.to_csv(os.path.join(output_dir, "df_test.csv"))
        df_sorted_all.to_csv(os.path.join(output_dir, "df_sorted_all.csv"))
        print(f"\nData saved to {output_dir}")

    return this_round_variants, df_test, df_sorted_all


def evolve_experimental_multi(
    protein_name: str,
    round_name: str,
    embeddings_base_path: str,
    embeddings_file_names: List[str],
    round_base_path: str,
    round_file_names_single: List[str],
    round_file_names_multi: List[str],
    wt_fasta_path: str,
    rename_WT: bool = False,
    number_of_variants: int = 12,
    output_dir: str = "/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/exp_results/",
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    bootstrap_sample_frac: float = DEFAULT_BOOTSTRAP_SAMPLE_FRAC,
    ucb_beta: float = DEFAULT_UCB_BETA,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Processing {protein_name} - {round_name}")

    embeddings_list = []
    for i, file_name in enumerate(embeddings_file_names):
        embedding = load_experimental_embeddings(
            embeddings_base_path, file_name, rename_WT
        )
        if i > 0:
            embedding = embedding[
                ~embedding.index.isin(["WT", "WT Wild-type sequence"])
            ]
        embeddings_list.append(embedding)

    embeddings = pd.concat(embeddings_list)
    embeddings = _prepare_plm_block(embeddings)
    print(f"Embeddings loaded: {embeddings.shape}")

    all_experimental_data = []
    for round_file_name in round_file_names_single:
        experimental_data = load_experimental_data(
            round_base_path, round_file_name, wt_fasta_path, single_mutant=True
        )
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")

    for round_file_name in round_file_names_multi:
        experimental_data = load_experimental_data(
            round_base_path, round_file_name, wt_fasta_path, single_mutant=False
        )
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")

    iteration, labels = create_iteration_dataframes(
        all_experimental_data, embeddings.index
    )

    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration["iteration"].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var="activity",
        regression_type="randomforest",
        experimental=True,
        ensemble_size=ensemble_size,
        bootstrap_sample_frac=bootstrap_sample_frac,
        ucb_beta=ucb_beta,
        enable_tail_weighting=enable_tail_weighting,
        tail_weight_mode=tail_weight_mode,
        tail_weight_quantile=tail_weight_quantile,
        tail_weight_strength=tail_weight_strength,
        tail_weight_max=tail_weight_max,
        enable_head_rerank=enable_head_rerank,
        head_rerank_pool_size=head_rerank_pool_size,
        head_rerank_weight=head_rerank_weight,
        head_rerank_tail_quantile=head_rerank_tail_quantile,
        enable_success_head=enable_success_head,
        success_label_mode=success_label_mode,
        success_threshold=success_threshold,
        success_positive_quantile=success_positive_quantile,
    )

    pred_col = "pred_mean" if "pred_mean" in df_test.columns else "y_pred"

    print(f"\nTested variants in this round: {len(this_round_variants)}")
    print(this_round_variants)
    print(f"\nTop {number_of_variants} variants predicted by the model:")
    print(
        df_test.sort_values(by=[pred_col], ascending=False).head(number_of_variants)[
            ["variant", pred_col]
        ]
    )

    if output_dir is not None:
        output_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, "iteration.csv"))
        this_round_variants.to_csv(os.path.join(output_dir, "this_round_variants.csv"))
        df_test = df_test.sort_values(by=[pred_col], ascending=False)
        df_test.to_csv(os.path.join(output_dir, "df_test.csv"))
        df_sorted_all.to_csv(os.path.join(output_dir, "df_sorted_all.csv"))
        print(f"\nData saved to {output_dir}")

    return this_round_variants, df_test, df_sorted_all
