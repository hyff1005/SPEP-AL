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

BASE_DIR=${BASE_DIR:-/public/home/huoyifan_pfr/code/EvolvePro}

SCRIPT=${SCRIPT:-${BASE_DIR}/scripts/struct/dms_main.py}

EMB_DIR=${EMB_DIR:-${BASE_DIR}/output/plm/esm}
LBL_DIR=${LBL_DIR:-${BASE_DIR}/output/dms}

STRUCT19_DIR=${STRUCT19_DIR:-${BASE_DIR}/output/struct}

WT11_DIR=${WT11_DIR:-${BASE_DIR}/output/struct_ablation/wt11}

OUT_ROOT=${OUT_ROOT:-${BASE_DIR}/output/formal_experiments}
LOG_ROOT=${LOG_ROOT:-${OUT_ROOT}/_logs}

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH:-}

MODEL_NAME="esm2_t36_3B_UR50D"
EMB_FILE_TYPE="csv"

NUM_SIM=${NUM_SIM:-10}
NUM_ITER=${NUM_ITER:-10}
NUM_MUT=${NUM_MUT:-16}
NUM_FINAL=${NUM_FINAL:-16}

MEASURED_VAR="activity"
LEARNING_STRATEGY="position_entropy_pareto"
FIRST_ROUND_STRATEGY="random"

EMBEDDING_TYPE="embeddings_pca_10_struct_fusion"
REGRESSION_TYPE="randomforest"

FORCE_RERUN=${FORCE_RERUN:-0}

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

EXP_NAME="finalC_pca10_wt11_randomforest_position_entropy_pareto_activity_sim${NUM_SIM}_iter${NUM_ITER}"

RESULT_DIR="${OUT_ROOT}/${EXP_NAME}"
LOG_DIR="${LOG_ROOT}/${EXP_NAME}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${WT11_DIR}"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

command -v python3 >/dev/null 2>&1 || die "python3 不存在。"
[ -f "${SCRIPT}" ] || die "dms_main.py 不存在：${SCRIPT}"
[ -d "${EMB_DIR}" ] || die "embedding目录不存在：${EMB_DIR}"
[ -d "${LBL_DIR}" ] || die "labels目录不存在：${LBL_DIR}"
[ -d "${STRUCT19_DIR}" ] || die "Struct19目录不存在：${STRUCT19_DIR}"

PRECHECK_FAILED=0

for dataset_name in "${DATASETS[@]}"; do
    emb_file="${EMB_DIR}/${dataset_name}_${MODEL_NAME}.${EMB_FILE_TYPE}"
    label_file="${LBL_DIR}/${dataset_name}_labels.csv"
    struct_file="${STRUCT19_DIR}/${dataset_name}_struct.csv"

    [ -f "${emb_file}" ] || {
        echo "[ERROR] 缺少嵌入：${emb_file}"
        PRECHECK_FAILED=1
    }

    [ -f "${label_file}" ] || {
        echo "[ERROR] 缺少标签：${label_file}"
        PRECHECK_FAILED=1
    }

    [ -f "${struct_file}" ] || {
        echo "[ERROR] 缺少Struct19：${struct_file}"
        PRECHECK_FAILED=1
    }
done

[ "${PRECHECK_FAILED}" -eq 0 ] || die "输入文件预检查失败。"

echo "============================================================"
echo "[PREP] 从Struct19重新构造WT11输入"
echo "============================================================"

export STRUCT19_DIR WT11_DIR
export DATASETS_JOINED="${DATASETS[*]}"

python3 - <<'PY'
import os
import shutil
from pathlib import Path

import pandas as pd

src_dir = Path(os.environ["STRUCT19_DIR"])
dst_dir = Path(os.environ["WT11_DIR"])
datasets = os.environ["DATASETS_JOINED"].split()

dst_dir.mkdir(parents=True, exist_ok=True)

WT11_COLS = [
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
]

summary = []

for dataset in datasets:
    src_struct = src_dir / f"{dataset}_struct.csv"
    dst_struct = dst_dir / f"{dataset}_struct.csv"

    df = pd.read_csv(src_struct, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    df = df[~df.index.duplicated(keep="first")]

    missing = [c for c in WT11_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{dataset}: Struct19缺少WT11列：{missing}"
        )

    out = df[WT11_COLS].copy()
    out.to_csv(dst_struct)

    src_qc = src_dir / f"{dataset}_struct_qc.csv"
    dst_qc = dst_dir / f"{dataset}_struct_qc.csv"

    if src_qc.exists():
        shutil.copy2(src_qc, dst_qc)
        qc_available = True
    else:
        qc_available = False

    summary.append(
        {
            "dataset": dataset,
            "rows": len(out),
            "features": out.shape[1],
            "qc_available": qc_available,
            "output": str(dst_struct),
        }
    )

summary_df = pd.DataFrame(summary)
summary_file = dst_dir / "wt11_prepare_summary_final.csv"
summary_df.to_csv(summary_file, index=False)

print(summary_df.to_string(index=False))
print(f"[DONE] WT11准备完成：{dst_dir}")
PY

echo "============================================================"
echo "[INFO] Experiment             : ${EXP_NAME}"
echo "[INFO] dms_main.py            : ${SCRIPT}"
echo "[INFO] Datasets               : ${#DATASETS[@]}"
echo "[INFO] Simulations            : ${NUM_SIM}"
echo "[INFO] Iterations             : ${NUM_ITER}"
echo "[INFO] Mutants per round      : ${NUM_MUT}"
echo "[INFO] Final mutants          : ${NUM_FINAL}"
echo "[INFO] Measured variable      : ${MEASURED_VAR}"
echo "[INFO] Embedding type         : ${EMBEDDING_TYPE}"
echo "[INFO] Structural features    : WT11 only"
echo "[INFO] WT11 path              : ${WT11_DIR}"
echo "[INFO] Regression type        : ${REGRESSION_TYPE}"
echo "[INFO] Learning strategy      : ${LEARNING_STRATEGY}"
echo "[INFO] Pareto pool multiplier : 4.0"
echo "[INFO] Quality retention      : 0.95"
echo "[INFO] Balance rule           : product"
echo "[INFO] Result directory       : ${RESULT_DIR}"
echo "[INFO] Log directory          : ${LOG_DIR}"
echo "[INFO] FORCE_RERUN            : ${FORCE_RERUN}"
echo "============================================================"

FAILED_DATASETS=()
TOTAL_RUN=0
TOTAL_SKIP=0

for dataset_name in "${DATASETS[@]}"; do
    result_file="${RESULT_DIR}/${dataset_name}_${MODEL_NAME}_${EXP_NAME}.csv"
    log_file="${LOG_DIR}/${dataset_name}.out"
    time_file="${LOG_DIR}/${dataset_name}.time.txt"

    if [ "${FORCE_RERUN}" -ne 1 ] && [ -s "${result_file}" ]; then
        echo "[SKIP] 已有结果：${dataset_name}"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[RUN] Dataset: ${dataset_name}"
    echo "[RUN] Log    : ${log_file}"
    echo "------------------------------------------------------------"

    cmd=(
        python3 -u "${SCRIPT}"

        --dataset_name "${dataset_name}"
        --experiment_name "${EXP_NAME}"
        --model_name "${MODEL_NAME}"

        --embeddings_path "${EMB_DIR}"
        --labels_path "${LBL_DIR}"
        --struct_path "${WT11_DIR}"

        --num_simulations "${NUM_SIM}"
        --num_iterations "${NUM_ITER}"
        --measured_var "${MEASURED_VAR}"
        --learning_strategies "${LEARNING_STRATEGY}"
        --num_mutants_per_round "${NUM_MUT}"
        --num_final_round_mutants "${NUM_FINAL}"
        --first_round_strategies "${FIRST_ROUND_STRATEGY}"

        --embedding_types "${EMBEDDING_TYPE}"
        --pca_components 10
        --regression_types "${REGRESSION_TYPE}"

        --embeddings_file_type "${EMB_FILE_TYPE}"
        --output_dir "${RESULT_DIR}"

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

        --pareto_lambda_values \
            0.0 0.05 0.10 0.20 0.40 \
            0.80 1.20 1.60 2.00

        --pareto_balance_rule product
        --pareto_min_quality_retention 0.95
        --pareto_local_search true
        --pareto_local_search_steps 50
        --pareto_entropy_base batch
        --pareto_force_top1 true
    )

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
        echo "[FAIL] ${dataset_name}, exit=${exit_code}"
        echo "       查看日志：${log_file}"
        FAILED_DATASETS+=("${dataset_name}")
        continue
    fi

    if [ ! -s "${result_file}" ]; then
        echo "[FAIL] ${dataset_name} 命令返回成功，但结果文件不存在或为空。"
        echo "       预期结果：${result_file}"
        FAILED_DATASETS+=("${dataset_name}")
        continue
    fi

    line_count=$(wc -l < "${result_file}")

    if [ "${line_count}" -lt 2 ]; then
        echo "[FAIL] ${dataset_name} 结果文件行数异常：${line_count}"
        FAILED_DATASETS+=("${dataset_name}")
        continue
    fi

    echo "[DONE] ${dataset_name}, lines=${line_count}"
done

echo "============================================================"
echo "[SUMMARY] 新运行任务数：${TOTAL_RUN}"
echo "[SUMMARY] 跳过已有结果：${TOTAL_SKIP}"

if [ "${#FAILED_DATASETS[@]}" -gt 0 ]; then
    fail_file="${RESULT_DIR}/failed_datasets.txt"
    printf "%s\n" "${FAILED_DATASETS[@]}" > "${fail_file}"

    echo "[ERROR] 失败数据集：${FAILED_DATASETS[*]}"
    echo "[ERROR] 失败清单：${fail_file}"
    echo "[INFO] Logs：${LOG_DIR}"
    exit 1
fi

echo "[DONE] WT11 + Position-Entropy Pareto 全部完成"
echo "[INFO] Results：${RESULT_DIR}"
echo "[INFO] Logs   ：${LOG_DIR}"
echo "[NEXT] python summarize_wt11_pareto_final.py"
echo "============================================================"
