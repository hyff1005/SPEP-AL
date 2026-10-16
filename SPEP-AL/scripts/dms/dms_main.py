import argparse
from typing import Optional

from evolvepro.src.evolve import grid_search


# ============================================================
# CLI helpers
# ============================================================

def str2bool(v):
    """
    Robust boolean parser for CLI.
    Supports: true/false, yes/no, y/n, 1/0, on/off
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return None

    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if s in {"false", "f", "no", "n", "0", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def normalize_success_label_mode(mode: Optional[str]) -> str:
    """
    Keep old bash scripts compatible while matching model.py accepted modes.

    model.py accepts:
      activity_binary / binary / hit / success
      scaled_threshold / threshold / activity_scaled
      quantile / scaled_quantile / activity_quantile
      auto

    Older dms_main.py/bashes used names such as activity_scaled_threshold
    and train_quantile. They are normalized here before calling grid_search.
    """
    s = str(mode or "activity_binary").strip().lower()
    aliases = {
        "activity_binary": "activity_binary",
        "binary": "activity_binary",
        "hit": "activity_binary",
        "success": "activity_binary",
        "activity_scaled_threshold": "scaled_threshold",
        "scaled_threshold": "scaled_threshold",
        "activity_threshold": "threshold",
        "measured_threshold": "threshold",
        "threshold": "threshold",
        "activity_scaled": "activity_scaled",
        "train_quantile": "quantile",
        "quantile": "quantile",
        "scaled_quantile": "scaled_quantile",
        "activity_quantile": "activity_quantile",
        "auto": "auto",
    }
    if s not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unsupported success_label_mode: {mode}. "
            f"Supported modes: {', '.join(sorted(aliases))}"
        )
    return aliases[s]


# ============================================================
# Parser
# ============================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EVOLVEpro grid-search experiments with PLM/structure-enhanced "
            "representations and multiple batch-selection strategies, including "
            "prediction-position-entropy Pareto batch selection. Legacy uncertainty, "
            "success-aware, mixed, and guarded strategies remain available for "
            "backward compatibility."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # Basic experiment inputs
    # ============================================================
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model used for embeddings.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the embeddings file or directory.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to the labels file or directory.",
    )
    parser.add_argument(
        "--struct_path",
        type=str,
        default=None,
        help="Path to the structural features directory.",
    )

    # ============================================================
    # Search space / simulation configuration
    # ============================================================
    parser.add_argument(
        "--num_simulations",
        type=int,
        required=True,
        help="Number of simulations for each parameter combination. Example: 3, 10, 30",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        nargs="+",
        required=True,
        help="List of number of iterations. Example: 3 5 10",
    )
    parser.add_argument(
        "--measured_var",
        type=str,
        nargs="+",
        required=True,
        help="Activity type to train on. Options: activity activity_scaled",
    )
    parser.add_argument(
        "--learning_strategies",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Learning strategies. Options include random, topn, dist, topn2bottomn2, "
            "position_entropy_pareto, ucb, ucb_diverse, mixed_ucb_diverse, "
            "mixed_ucb_diverse_schedule, thompson, success_topn, "
            "success_constrained_mixed, safe_struct_mixed, and guarded_struct_mixed. "
            "The proposed method is position_entropy_pareto."
        ),
    )
    parser.add_argument(
        "--num_mutants_per_round",
        type=int,
        nargs="+",
        required=True,
        help="Number of mutants per round. Example: 4 8 16 32 128",
    )
    parser.add_argument(
        "--num_final_round_mutants",
        type=int,
        required=True,
        help="Number of mutants used when computing final-round top statistics. Example: 16",
    )
    parser.add_argument(
        "--first_round_strategies",
        type=str,
        nargs="+",
        required=True,
        help="First round strategy. Options: random diverse_medoids explicit_variants",
    )
    parser.add_argument(
        "--embedding_types",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Types of embeddings to train on. Examples: embeddings embeddings_struct_only "
            "embeddings_struct_fusion embeddings_pca_32 embeddings_pca_32_struct_fusion"
        ),
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Legacy argument. PCA dimension is currently inferred from embedding_types "
            "such as embeddings_pca_32."
        ),
    )
    parser.add_argument(
        "--regression_types",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Regression types. Options: ridge lasso elasticnet linear neuralnet "
            "randomforest gradientboosting knn gp hybrid"
        ),
    )
    parser.add_argument(
        "--embeddings_file_type",
        type=str,
        required=True,
        help="Type of embeddings file to read. Options: csv pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for grid search results.",
    )
    parser.add_argument(
        "--embeddings_type_pt",
        type=str,
        default=None,
        help="Type of embeddings to use for .pt files. Options: average mutated both",
    )

    # ============================================================
    # Uncertainty-aware / acquisition parameters
    # ============================================================
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help=(
            "Number of bootstrap ensemble members used in the top-layer model. "
            "The proposed Pareto method uses a deterministic single model by default."
        ),
    )
    parser.add_argument(
        "--bootstrap_sample_frac",
        type=float,
        default=1.0,
        help="Fraction of training samples drawn with replacement for each bootstrap member.",
    )
    parser.add_argument(
        "--ucb_beta",
        type=float,
        default=0.0,
        help=(
            "Weight of predictive uncertainty in UCB-style acquisition. "
            "For success_topn / success_constrained_mixed / safe_struct_mixed, recommended value is 0.0."
        ),
    )
    parser.add_argument(
        "--novelty_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of novelty-to-tested bonus in UCB-style acquisition. "
            "For success-first or safe methods, keep this 0.0 and use the method-specific novelty parameters instead."
        ),
    )
    parser.add_argument(
        "--redundancy_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of diversity bonus when building a batch greedily. "
            "For success_constrained_mixed, recommended value is small, e.g. 0.03."
        ),
    )
    parser.add_argument(
        "--thompson_scale",
        type=float,
        default=1.0,
        help="Scale factor for Thompson-style sampling using pred_mean + noise * pred_std.",
    )
    parser.add_argument(
        "--mixed_exploit_frac",
        type=float,
        default=0.75,
        help=(
            "Default fraction of batch allocated to exploitation in mixed strategies. "
            "For success_constrained_mixed with batch size 16, 0.9375 means 15 exploit + 1 constrained explore. "
            "When mixed_exploit_schedule is provided, this value is fallback for uncovered rounds."
        ),
    )
    parser.add_argument(
        "--mixed_exploit_schedule",
        type=str,
        default=None,
        help=(
            "Optional round-dependent exploit schedule for mixed strategies. "
            "Supported forms: schedule_a, schedule_b, or explicit strings like "
            "'1-3:0.75,4-6:0.8125,7-10:0.875'."
        ),
    )

    # ============================================================
    # Tail-aware weighted top-layer parameters
    # ============================================================
    parser.add_argument(
        "--enable_tail_weighting",
        type=str2bool,
        default=False,
        help=(
            "Whether to enable tail-aware sample weighting in the top-layer model. "
            "Disabled by default for the structure-plus-Pareto main experiment."
        ),
    )
    parser.add_argument(
        "--tail_weight_mode",
        type=str,
        default="quantile_linear",
        help="Tail-weighting mode for top-layer fitting. Recommended: quantile_linear",
    )
    parser.add_argument(
        "--tail_weight_quantile",
        type=float,
        default=0.80,
        help="Quantile threshold above which training samples receive stronger tail-aware weights.",
    )
    parser.add_argument(
        "--tail_weight_strength",
        type=float,
        default=2.0,
        help="Strength of tail-aware upweighting beyond the chosen quantile threshold.",
    )
    parser.add_argument(
        "--tail_weight_max",
        type=float,
        default=5.0,
        help="Maximum cap applied to tail-aware sample weights.",
    )

    # ============================================================
    # Lightweight head rerank parameters
    # ============================================================
    parser.add_argument(
        "--enable_head_rerank",
        type=str2bool,
        default=False,
        help=(
            "Whether to apply lightweight head reranking on the top candidate pool. "
            "Disabled by default for the structure-plus-Pareto main experiment."
        ),
    )
    parser.add_argument(
        "--head_rerank_pool_size",
        type=int,
        default=64,
        help="Size of the candidate head pool used for lightweight reranking.",
    )
    parser.add_argument(
        "--head_rerank_weight",
        type=float,
        default=0.15,
        help="Strength of the rerank correction added to pred_mean within the head pool.",
    )
    parser.add_argument(
        "--head_rerank_tail_quantile",
        type=float,
        default=0.80,
        help="Tail quantile used to define high-activity anchors for lightweight head reranking.",
    )

    # ============================================================
    # Success-aware two-head parameters
    # ============================================================
    parser.add_argument(
        "--enable_success_head",
        type=str2bool,
        default=False,
        help=(
            "Whether to train an additional success-probability head. "
            "When enabled, model.py outputs pred_success_prob / pred_success_prob_std. "
            "Required for true success-first selection."
        ),
    )
    parser.add_argument(
        "--success_label_mode",
        type=str,
        default="activity_binary",
        help=(
            "How to construct the binary success label. Recommended: activity_binary. "
            "Compatible modes: activity_binary, scaled_threshold, threshold, quantile, auto. "
            "Old aliases are also accepted: activity_scaled_threshold, activity_threshold, "
            "measured_threshold, train_quantile."
        ),
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.50,
        help="Threshold used when success_label_mode is scaled_threshold / threshold.",
    )
    parser.add_argument(
        "--success_positive_quantile",
        type=float,
        default=0.70,
        help="Training-set quantile used when success_label_mode is quantile / train_quantile.",
    )
    parser.add_argument(
        "--success_weight",
        type=float,
        default=0.0,
        help=(
            "Legacy weight of pred_success_prob rank in ucb/mixed acquisition. "
            "For success_topn / success_constrained_mixed, use the success_acq_* weights instead."
        ),
    )
    parser.add_argument(
        "--success_filter_quantile",
        type=float,
        default=0.0,
        help=(
            "Legacy filter. In filter_then_score mode, remove candidates below this quantile of "
            "pred_success_prob before selection. Example: 0.2 removes the lowest 20%."
        ),
    )
    parser.add_argument(
        "--success_mode",
        type=str,
        default="score_bonus",
        choices=["off", "score_bonus", "filter_then_score", "filter", "filter_then_ucb"],
        help=(
            "How legacy acquisition uses pred_success_prob. score_bonus adds probability bonus; "
            "filter_then_score first removes low-probability candidates. "
            "success_topn / success_constrained_mixed use success-first scoring regardless."
        ),
    )

    # ============================================================
    # Success-first acquisition parameters
    # ============================================================
    parser.add_argument(
        "--success_acq_success_weight",
        type=float,
        default=0.75,
        help=(
            "Weight of rank(pred_success_prob) in success-first acquisition. "
            "This is the main term for success_topn / success_constrained_mixed."
        ),
    )
    parser.add_argument(
        "--success_acq_pred_weight",
        type=float,
        default=0.20,
        help="Weight of rank(pred_mean) in success-first acquisition.",
    )
    parser.add_argument(
        "--success_acq_std_penalty",
        type=float,
        default=0.10,
        help=(
            "Penalty weight of rank(pred_std) in success-first acquisition. "
            "Positive value means uncertainty is treated as risk and subtracted."
        ),
    )
    parser.add_argument(
        "--success_acq_novelty_weight",
        type=float,
        default=0.03,
        help=(
            "Small novelty bonus used only for success-first/constrained exploration. "
            "Keep small to avoid lowering hit rate."
        ),
    )
    parser.add_argument(
        "--success_explore_success_quantile",
        type=float,
        default=0.50,
        help=(
            "For success_constrained_mixed, explore candidates must be above this "
            "pred_success_prob quantile before diversity is considered."
        ),
    )
    parser.add_argument(
        "--success_explore_pred_quantile",
        type=float,
        default=0.50,
        help=(
            "For success_constrained_mixed, explore candidates must be above this "
            "pred_mean quantile before diversity is considered."
        ),
    )


    # ============================================================
    # Safe / guarded structure-aware acquisition parameters
    # ============================================================
    parser.add_argument(
        "--safe_max_replace",
        type=int,
        default=1,
        help=(
            "For safe_struct_mixed / guarded_struct_mixed, maximum number of tail TopN "
            "positions that can be replaced by guarded candidates. Recommended: 1 or 2."
        ),
    )
    parser.add_argument(
        "--safe_candidate_pool_quantile",
        type=float,
        default=0.80,
        help=(
            "For safe_struct_mixed / guarded_struct_mixed, candidate replacements are drawn "
            "only from variants above this pred_mean quantile. 0.80 means top 20% by pred_mean."
        ),
    )
    parser.add_argument(
        "--safe_std_penalty",
        type=float,
        default=0.10,
        help=(
            "Risk penalty applied to rank(pred_std) in safe acquisition. "
            "Positive values penalize uncertain candidates."
        ),
    )
    parser.add_argument(
        "--safe_novelty_weight",
        type=float,
        default=0.01,
        help=(
            "Small novelty bonus for guarded replacement candidates. Keep small, e.g. 0.00-0.02, "
            "to avoid reducing success rate."
        ),
    )
    parser.add_argument(
        "--safe_batch_margin",
        type=float,
        default=0.00,
        help=(
            "Batch-level guard margin. A replacement batch is accepted only if its safe batch score "
            "is at least TopN batch score minus this margin. Use 0.0 for strict no-regret guarding."
        ),
    )

    parser.add_argument(
        "--safe_require_batch_score",
        type=str2bool,
        default=True,
        help=(
            "Whether safe_struct_mixed / guarded_struct_mixed should enforce the batch-level "
            "guard and fall back to Struct TopN when the guarded batch score is worse."
        ),
    )

    # ============================================================
    # Prediction-position-entropy Pareto batch selection
    # ============================================================
    parser.add_argument(
        "--pareto_pool_multiplier",
        type=float,
        default=4.0,
        help=(
            "Multiplier used to construct the high-prediction candidate pool. "
            "For batch size N, pool size K is ceil(multiplier * N), capped by the "
            "number of available untested variants. Example: 4.0 gives Top-64 "
            "when N=16."
        ),
    )
    parser.add_argument(
        "--pareto_lambda_values",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00],
        help=(
            "Non-negative trade-off coefficients used to generate candidate batches. "
            "Each lambda searches a batch maximizing normalized prediction quality "
            "+ lambda * normalized mutation-position entropy."
        ),
    )
    parser.add_argument(
        "--pareto_balance_rule",
        type=str,
        default="product",
        choices=["product", "knee", "max_entropy_under_quality"],
        help=(
            "Rule for selecting one batch from the non-dominated Pareto frontier. "
            "product maximizes normalized quality times normalized entropy; knee "
            "selects the geometric knee; max_entropy_under_quality selects the "
            "highest-entropy batch satisfying pareto_min_quality_retention."
        ),
    )
    parser.add_argument(
        "--pareto_min_quality_retention",
        type=float,
        default=0.95,
        help=(
            "Minimum retained mean prediction quality relative to the deterministic "
            "Top-N batch. Used as a safety constraint when choosing the final Pareto "
            "batch. Set 0 to disable; recommended value is 0.95."
        ),
    )
    parser.add_argument(
        "--pareto_local_search",
        type=str2bool,
        default=True,
        help=(
            "Whether to refine each greedily generated Pareto batch with deterministic "
            "one-for-one swap local search."
        ),
    )
    parser.add_argument(
        "--pareto_local_search_steps",
        type=int,
        default=50,
        help=(
            "Maximum number of accepted/improvement search steps for each lambda. "
            "Use 0 to disable refinement even when pareto_local_search is true."
        ),
    )
    parser.add_argument(
        "--pareto_entropy_base",
        type=str,
        default="batch",
        choices=["batch", "available_positions"],
        help=(
            "Normalization denominator for mutation-position entropy. batch uses "
            "log(N), so entropy reaches 1 when all N selected variants have distinct "
            "positions. available_positions uses log(min(N, unique positions in the "
            "candidate pool))."
        ),
    )
    parser.add_argument(
        "--pareto_force_top1",
        type=str2bool,
        default=True,
        help=(
            "Whether every Pareto batch must contain the highest-predicted candidate "
            "from the high-prediction pool. This preserves a clear exploitation anchor."
        ),
    )

    return parser


# ============================================================
# Validation
# ============================================================

def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate only cross-argument constraints that argparse cannot express."""
    if args.ensemble_size < 1:
        parser.error("--ensemble_size must be >= 1")
    if not (0.0 < args.bootstrap_sample_frac <= 1.0):
        parser.error("--bootstrap_sample_frac must be in (0, 1]")

    if args.pareto_pool_multiplier < 1.0:
        parser.error("--pareto_pool_multiplier must be >= 1.0")
    if not args.pareto_lambda_values:
        parser.error("--pareto_lambda_values must contain at least one value")
    if any(value < 0.0 for value in args.pareto_lambda_values):
        parser.error("--pareto_lambda_values must all be non-negative")
    if not (0.0 <= args.pareto_min_quality_retention <= 1.0):
        parser.error("--pareto_min_quality_retention must be in [0, 1]")
    if args.pareto_local_search_steps < 0:
        parser.error("--pareto_local_search_steps must be >= 0")

    if any(value < 1 for value in args.num_mutants_per_round):
        parser.error("All --num_mutants_per_round values must be >= 1")
    if any(value < 1 for value in args.num_iterations):
        parser.error("All --num_iterations values must be >= 1")

    # Keep lambda order deterministic and remove exact duplicates while preserving order.
    args.pareto_lambda_values = list(dict.fromkeys(float(v) for v in args.pareto_lambda_values))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    success_label_mode = normalize_success_label_mode(args.success_label_mode)

    grid_search(
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        embeddings_path=args.embeddings_path,
        labels_path=args.labels_path,
        num_simulations=args.num_simulations,
        num_iterations=args.num_iterations,
        measured_var=args.measured_var,
        learning_strategies=args.learning_strategies,
        num_mutants_per_round=args.num_mutants_per_round,
        num_final_round_mutants=args.num_final_round_mutants,
        first_round_strategies=args.first_round_strategies,
        embedding_types=args.embedding_types,
        pca_components=args.pca_components,
        regression_types=args.regression_types,
        embeddings_file_type=args.embeddings_file_type,
        output_dir=args.output_dir,
        embeddings_type_pt=args.embeddings_type_pt,
        struct_path=args.struct_path,
        ensemble_size=args.ensemble_size,
        bootstrap_sample_frac=args.bootstrap_sample_frac,
        ucb_beta=args.ucb_beta,
        novelty_weight=args.novelty_weight,
        redundancy_weight=args.redundancy_weight,
        thompson_scale=args.thompson_scale,
        mixed_exploit_frac=args.mixed_exploit_frac,
        mixed_exploit_schedule=args.mixed_exploit_schedule,
        enable_tail_weighting=args.enable_tail_weighting,
        tail_weight_mode=args.tail_weight_mode,
        tail_weight_quantile=args.tail_weight_quantile,
        tail_weight_strength=args.tail_weight_strength,
        tail_weight_max=args.tail_weight_max,
        enable_head_rerank=args.enable_head_rerank,
        head_rerank_pool_size=args.head_rerank_pool_size,
        head_rerank_weight=args.head_rerank_weight,
        head_rerank_tail_quantile=args.head_rerank_tail_quantile,
        enable_success_head=args.enable_success_head,
        success_label_mode=success_label_mode,
        success_threshold=args.success_threshold,
        success_positive_quantile=args.success_positive_quantile,
        success_weight=args.success_weight,
        success_filter_quantile=args.success_filter_quantile,
        success_mode=args.success_mode,
        success_acq_success_weight=args.success_acq_success_weight,
        success_acq_pred_weight=args.success_acq_pred_weight,
        success_acq_std_penalty=args.success_acq_std_penalty,
        success_acq_novelty_weight=args.success_acq_novelty_weight,
        success_explore_success_quantile=args.success_explore_success_quantile,
        success_explore_pred_quantile=args.success_explore_pred_quantile,
        safe_max_replace=args.safe_max_replace,
        safe_candidate_pool_quantile=args.safe_candidate_pool_quantile,
        safe_std_penalty=args.safe_std_penalty,
        safe_novelty_weight=args.safe_novelty_weight,
        safe_batch_margin=args.safe_batch_margin,
        safe_require_batch_score=args.safe_require_batch_score,
        pareto_pool_multiplier=args.pareto_pool_multiplier,
        pareto_lambda_values=args.pareto_lambda_values,
        pareto_balance_rule=args.pareto_balance_rule,
        pareto_min_quality_retention=args.pareto_min_quality_retention,
        pareto_local_search=args.pareto_local_search,
        pareto_local_search_steps=args.pareto_local_search_steps,
        pareto_entropy_base=args.pareto_entropy_base,
        pareto_force_top1=args.pareto_force_top1,
    )


if __name__ == "__main__":
    main()
