#!/bin/bash
set -eo pipefail


export PS1="${PS1-}"

set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] 找不到 conda.sh，请检查 conda 安装路径。"
    exit 1
fi

conda activate plm
set -u

# ------------------------------------------------------------
# 路径与固定配置
# ------------------------------------------------------------
BASE_DIR=${BASE_DIR:-/code/EvolvePro}
SCRIPT=${SCRIPT:-${BASE_DIR}/scripts/struct/dms_main.py}

EMB_DIR=${EMB_DIR:-${BASE_DIR}/output/plm/esm}
LBL_DIR=${LBL_DIR:-${BASE_DIR}/output/dms}
STRUCT_DIR=${STRUCT_DIR:-${BASE_DIR}/output/struct}

OUT_ROOT=${OUT_ROOT:-${BASE_DIR}/output/formal_experiments}
LOG_ROOT=${LOG_ROOT:-${OUT_ROOT}/_logs}

FINAL_ABLATION_ROOT=${FINAL_ABLATION_ROOT:-${BASE_DIR}/output/struct_ablation_final}

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH:-}

MODEL_NAME="esm2_t36_3B_UR50D"
EMB_FILE_TYPE="csv"

NUM_SIM=${NUM_SIM:-10}
NUM_ITER=${NUM_ITER:-10}
NUM_MUT=${NUM_MUT:-16}
NUM_FINAL=${NUM_FINAL:-16}

MEASURED_VAR="activity"
FIRST_ROUND_STRATEGY="random"
REGRESSION_TYPE="randomforest"

FORCE_RERUN=${FORCE_RERUN:-0}

DEFAULT_LAMBDAS="0.0 0.05 0.10 0.20 0.40 0.80 1.20 1.60 2.00"

DATASETS=(
    "brenan"
    "stiffler"
    "doud"
    "haddox"
    "giacomelli"
    "jones"
    "kelsic"
    "lee"
    "markin"
    "zikv_E"
    "cas12f"
    "cov2_S"
)

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# ------------------------------------------------------------
# 预检查
# ------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 不存在。"
[ -f "${SCRIPT}" ] || die "dms_main.py 不存在：${SCRIPT}"
[ -d "${EMB_DIR}" ] || die "嵌入目录不存在：${EMB_DIR}"
[ -d "${LBL_DIR}" ] || die "标签目录不存在：${LBL_DIR}"
[ -d "${STRUCT_DIR}" ] || die "结构目录不存在：${STRUCT_DIR}"

PRECHECK_FAILED=0

for dataset_name in "${DATASETS[@]}"; do
    emb_file="${EMB_DIR}/${dataset_name}_${MODEL_NAME}.${EMB_FILE_TYPE}"
    label_file="${LBL_DIR}/${dataset_name}_labels.csv"
    struct_file="${STRUCT_DIR}/${dataset_name}_struct.csv"

    [ -f "${emb_file}" ] || {
        echo "[ERROR] 缺少嵌入：${emb_file}"
        PRECHECK_FAILED=1
    }

    [ -f "${label_file}" ] || {
        echo "[ERROR] 缺少标签：${label_file}"
        PRECHECK_FAILED=1
    }

    [ -f "${struct_file}" ] || {
        echo "[ERROR] 缺少原始Struct19：${struct_file}"
        PRECHECK_FAILED=1
    }
done

[ "${PRECHECK_FAILED}" -eq 0 ] || die "输入文件预检查失败。"

mkdir -p \
    "${OUT_ROOT}" \
    "${LOG_ROOT}" \
    "${FINAL_ABLATION_ROOT}/position1" \
    "${FINAL_ABLATION_ROOT}/physchem3"

echo "============================================================"
echo "[PREP] 构造 position1 / physchem3 控制特征"
echo "============================================================"

export LBL_DIR FINAL_ABLATION_ROOT
export DATASETS_JOINED="${DATASETS[*]}"

python3 - <<'PY'
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

label_dir = Path(os.environ["LBL_DIR"])
out_root = Path(os.environ["FINAL_ABLATION_ROOT"])
datasets = os.environ["DATASETS_JOINED"].split()

position_dir = out_root / "position1"
physchem_dir = out_root / "physchem3"
position_dir.mkdir(parents=True, exist_ok=True)
physchem_dir.mkdir(parents=True, exist_ok=True)

AA_PROPS = {
    "A": (88.6, 1.8, 0.0),
    "R": (173.4, -4.5, 1.0),
    "N": (114.1, -3.5, 0.0),
    "D": (111.1, -3.5, -1.0),
    "C": (108.5, 2.5, 0.0),
    "Q": (143.8, -3.5, 0.0),
    "E": (138.4, -3.5, -1.0),
    "G": (60.1, -0.4, 0.0),
    "H": (153.2, -3.2, 0.1),
    "I": (166.7, 4.5, 0.0),
    "L": (166.7, 3.8, 0.0),
    "K": (168.6, -3.9, 1.0),
    "M": (162.9, 1.9, 0.0),
    "F": (189.9, 2.8, 0.0),
    "P": (112.7, -1.6, 0.0),
    "S": (89.0, -0.8, 0.0),
    "T": (116.1, -0.7, 0.0),
    "W": (227.8, -0.9, 0.0),
    "Y": (193.6, -1.3, 0.0),
    "V": (140.0, 4.2, 0.0),
}

pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")
summary_rows = []

for dataset in datasets:
    label_file = label_dir / f"{dataset}_labels.csv"
    labels = pd.read_csv(label_file)

    if "variant" not in labels.columns:
        raise RuntimeError(f"{dataset}: labels 缺少 variant 列")

    variants = (
        labels["variant"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    position_rows = []
    physchem_rows = []
    unparsed = []

    for variant in variants:
        v = str(variant).strip().upper()

        # WT 不属于单点突变，作为中性控制编码为0。
        if v in {"WT", "WILDTYPE", "WILD_TYPE"}:
            position_rows.append((variant, 0.0))
            physchem_rows.append((variant, 0.0, 0.0, 0.0))
            continue

        match = pattern.fullmatch(v)
        if match is None:
            unparsed.append(variant)
            position_rows.append((variant, np.nan))
            physchem_rows.append((variant, np.nan, np.nan, np.nan))
            continue

        wt, pos, mt = match.group(1), int(match.group(2)), match.group(3)

        if wt not in AA_PROPS or mt not in AA_PROPS:
            unparsed.append(variant)
            position_rows.append((variant, float(pos)))
            physchem_rows.append((variant, np.nan, np.nan, np.nan))
            continue

        wt_prop = AA_PROPS[wt]
        mt_prop = AA_PROPS[mt]

        position_rows.append((variant, float(pos)))
        physchem_rows.append(
            (
                variant,
                float(mt_prop[0] - wt_prop[0]),
                float(mt_prop[1] - wt_prop[1]),
                float(mt_prop[2] - wt_prop[2]),
            )
        )

    position_df = pd.DataFrame(
        position_rows,
        columns=["variant", "mutation_position"],
    ).set_index("variant")

    physchem_df = pd.DataFrame(
        physchem_rows,
        columns=[
            "variant",
            "delta_aa_volume",
            "delta_hydrophobicity",
            "delta_charge",
        ],
    ).set_index("variant")

    position_df.to_csv(position_dir / f"{dataset}_struct.csv")
    physchem_df.to_csv(physchem_dir / f"{dataset}_struct.csv")

    summary_rows.append(
        {
            "dataset": dataset,
            "n_variants": len(variants),
            "position_dim": 1,
            "physchem_dim": 3,
            "unparsed_non_wt": len(unparsed),
            "unparsed_examples": ",".join(map(str, unparsed[:10])),
        }
    )

summary = pd.DataFrame(summary_rows)
summary_file = out_root / "final_control_feature_prepare_summary.csv"
summary.to_csv(summary_file, index=False)

print(summary.to_string(index=False))
print(f"[DONE] 控制特征准备完成：{summary_file}")

if int(summary["unparsed_non_wt"].sum()) > 0:
    print(
        "[WARN] 存在无法按A123V解析的非WT variant。"
        "对应控制特征将作为NaN交给data.py中位数插补。"
    )
PY

EXP_NAMES=()
EXP_STRUCT_PATHS=()
EXP_EMBEDDINGS=()
EXP_STRATEGIES=()
EXP_LAMBDAS=()
EXP_RULES=()
EXP_QRETS=()
EXP_DESCRIPTIONS=()

add_exp() {
    EXP_NAMES+=("$1")
    EXP_STRUCT_PATHS+=("$2")
    EXP_EMBEDDINGS+=("$3")
    EXP_STRATEGIES+=("$4")
    EXP_LAMBDAS+=("$5")
    EXP_RULES+=("$6")
    EXP_QRETS+=("$7")
    EXP_DESCRIPTIONS+=("$8")
}

# A1. Position only
add_exp \
    "finalA_pca10_position1_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
    "${FINAL_ABLATION_ROOT}/position1" \
    "embeddings_pca_10_struct_fusion" \
    "topn" \
    "${DEFAULT_LAMBDAS}" \
    "product" \
    "0.95" \
    "PCA10 + mutation_position(1D) + RF + TopN"

# A2. Physchem only
add_exp \
    "finalA_pca10_physchem3_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
    "${FINAL_ABLATION_ROOT}/physchem3" \
    "embeddings_pca_10_struct_fusion" \
    "topn" \
    "${DEFAULT_LAMBDAS}" \
    "product" \
    "0.95" \
    "PCA10 + amino-acid physicochemical delta(3D) + RF + TopN"

for lambda_spec in \
    "010:0.10" \
    "040:0.40" \
    "080:0.80" \
    "120:1.20"
do
    lambda_tag="${lambda_spec%%:*}"
    lambda_value="${lambda_spec##*:}"

    add_exp \
        "finalB_struct19_singlelambda${lambda_tag}_q95_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${lambda_value}" \
        "max_entropy_under_quality" \
        "0.95" \
        "Struct19 + single lambda=${lambda_value} + q95 + RF"
done

MANIFEST=${OUT_ROOT}/final_supplement_manifest.tsv

{
    printf "experiment\tstruct_path\tembedding_type\tlearning_strategy\tlambda_values\tbalance_rule\tquality_retention\tdescription\n"

    for i in "${!EXP_NAMES[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${EXP_NAMES[$i]}" \
            "${EXP_STRUCT_PATHS[$i]}" \
            "${EXP_EMBEDDINGS[$i]}" \
            "${EXP_STRATEGIES[$i]}" \
            "${EXP_LAMBDAS[$i]}" \
            "${EXP_RULES[$i]}" \
            "${EXP_QRETS[$i]}" \
            "${EXP_DESCRIPTIONS[$i]}"
    done
} > "${MANIFEST}"

echo "============================================================"
echo "[INFO] 实验数量             : ${#EXP_NAMES[@]}"
echo "[INFO] 数据集数量           : ${#DATASETS[@]}"
echo "[INFO] simulations          : ${NUM_SIM}"
echo "[INFO] iterations           : ${NUM_ITER}"
echo "[INFO] mutants/round        : ${NUM_MUT}"
echo "[INFO] FORCE_RERUN          : ${FORCE_RERUN}"
echo "[INFO] manifest             : ${MANIFEST}"
echo "============================================================"

FAILED_ITEMS=()
TOTAL_RUN=0
TOTAL_SKIP=0

run_one() {
    local exp_name="$1"
    local struct_path="$2"
    local embedding_type="$3"
    local strategy="$4"
    local lambda_string="$5"
    local rule="$6"
    local qret="$7"
    local dataset_name="$8"

    local result_dir="${OUT_ROOT}/${exp_name}"
    local log_dir="${LOG_ROOT}/${exp_name}"
    local result_file="${result_dir}/${dataset_name}_${MODEL_NAME}_${exp_name}.csv"
    local log_file="${log_dir}/${dataset_name}.out"
    local time_file="${log_dir}/${dataset_name}.time.txt"

    mkdir -p "${result_dir}" "${log_dir}"

    if [ "${FORCE_RERUN}" -ne 1 ] && [ -s "${result_file}" ]; then
        echo "[SKIP] ${exp_name} / ${dataset_name}"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi

    read -r -a lambda_arr <<< "${lambda_string}"

    cmd=(
        python3 -u "${SCRIPT}"

        --dataset_name "${dataset_name}"
        --experiment_name "${exp_name}"
        --model_name "${MODEL_NAME}"

        --embeddings_path "${EMB_DIR}"
        --labels_path "${LBL_DIR}"
        --struct_path "${struct_path}"

        --num_simulations "${NUM_SIM}"
        --num_iterations "${NUM_ITER}"
        --measured_var "${MEASURED_VAR}"
        --learning_strategies "${strategy}"
        --num_mutants_per_round "${NUM_MUT}"
        --num_final_round_mutants "${NUM_FINAL}"
        --first_round_strategies "${FIRST_ROUND_STRATEGY}"

        --embedding_types "${embedding_type}"
        --pca_components 10
        --regression_types "${REGRESSION_TYPE}"

        --embeddings_file_type "${EMB_FILE_TYPE}"
        --output_dir "${result_dir}"

        --ensemble_size 1
        --bootstrap_sample_frac 1.0

        --ucb_beta 0.0
        --novelty_weight 0.0
        --redundancy_weight 0.0
        --thompson_scale 1.0
        --mixed_exploit_frac 1.0

        --enable_tail_weighting false
        --enable_head_rerank false
        --enable_success_head false
        --success_weight 0.0
        --success_filter_quantile 0.0
        --success_mode off

        --pareto_pool_multiplier 4.0
        --pareto_lambda_values "${lambda_arr[@]}"
        --pareto_balance_rule "${rule}"
        --pareto_min_quality_retention "${qret}"
        --pareto_local_search true
        --pareto_local_search_steps 50
        --pareto_entropy_base batch
        --pareto_force_top1 true
    )

    echo "------------------------------------------------------------"
    echo "[RUN] ${exp_name}"
    echo "      dataset   = ${dataset_name}"
    echo "      struct    = ${struct_path}"
    echo "      embedding = ${embedding_type}"
    echo "      strategy  = ${strategy}"
    echo "      lambda    = ${lambda_string}"
    echo "      rule/qret = ${rule}/${qret}"
    echo "      log       = ${log_file}"
    echo "------------------------------------------------------------"

    set +e

    if [ -x /usr/bin/time ]; then
        /usr/bin/time \
            -f "elapsed_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M" \
            -o "${time_file}" \
            "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
    else
        start_epoch=$(date +%s)
        "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
        end_epoch=$(date +%s)
        echo "elapsed_sec=$((end_epoch - start_epoch))" > "${time_file}"
    fi

    set -e

    TOTAL_RUN=$((TOTAL_RUN + 1))

    if [ "${exit_code}" -ne 0 ]; then
        echo "[FAIL] ${exp_name} / ${dataset_name}, exit=${exit_code}"
        echo "       查看日志：${log_file}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    if [ ! -s "${result_file}" ]; then
        echo "[FAIL] 结果文件不存在或为空：${result_file}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    line_count=$(wc -l < "${result_file}")
    if [ "${line_count}" -lt 2 ]; then
        echo "[FAIL] 结果文件行数异常：${result_file}, lines=${line_count}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    echo "[DONE] ${exp_name} / ${dataset_name}, lines=${line_count}"
}

# ------------------------------------------------------------
# 顺序运行
# ------------------------------------------------------------
for i in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$i]}"

    echo
    echo "############################################################"
    echo "[EXPERIMENT $((i + 1))/${#EXP_NAMES[@]}] ${exp_name}"
    echo "[DESC] ${EXP_DESCRIPTIONS[$i]}"
    echo "############################################################"

    for dataset_name in "${DATASETS[@]}"; do
        run_one \
            "${exp_name}" \
            "${EXP_STRUCT_PATHS[$i]}" \
            "${EXP_EMBEDDINGS[$i]}" \
            "${EXP_STRATEGIES[$i]}" \
            "${EXP_LAMBDAS[$i]}" \
            "${EXP_RULES[$i]}" \
            "${EXP_QRETS[$i]}" \
            "${dataset_name}"
    done
done

# ------------------------------------------------------------
# 完成汇总
# ------------------------------------------------------------
echo
echo "============================================================"
echo "[SUMMARY] 新运行任务数：${TOTAL_RUN}"
echo "[SUMMARY] 跳过已有结果：${TOTAL_SKIP}"
echo "[SUMMARY] Results：${OUT_ROOT}"
echo "[SUMMARY] Logs：${LOG_ROOT}"
echo "[SUMMARY] Manifest：${MANIFEST}"

if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
    FAIL_FILE="${OUT_ROOT}/final_supplement_failed_items.txt"
    printf "%s\n" "${FAILED_ITEMS[@]}" > "${FAIL_FILE}"

    echo "[ERROR] 失败任务数：${#FAILED_ITEMS[@]}"
    echo "[ERROR] 失败清单：${FAIL_FILE}"
    printf "  - %s\n" "${FAILED_ITEMS[@]}"
    exit 1
fi

echo "[DONE] 最后一轮补充实验全部完成。"
echo "[NEXT] 运行：python summarize_evolvepro_final_supplement.py"
echo "============================================================"
