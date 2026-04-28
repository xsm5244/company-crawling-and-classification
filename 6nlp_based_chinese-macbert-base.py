import os
import sys
import json
from pathlib import Path
import inspect

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


INPUT_CSV = r"E:\BaiduSyncdisk\企业科研\自动化分类\type_plan\政府采购_对应市_补全规划实际类别.csv"
LOCAL_MODEL_DIR = Path(r"E:\BaiduSyncdisk\企业科研\自动化分类\type_plan\hf_modelsw\chinese-macbert-base")

# =========================
# 1) 字段
# =========================
TEXT_COLS = ["company_name", "tenderee", "project_name"]
LABEL_COL = "规划实际类别"

# =========================
# 2) 稀有类兜底：< MIN_FREQ 的类别 => __RARE__
# =========================
MIN_FREQ = 2
RARE_LABEL = "__RARE__"

# =========================
# 3) 训练参数
# =========================
SEED = 42
TEST_SIZE = 0.15

MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
FP16 = True  # 无GPU可改 False

# =========================
# 4) 输出
# =========================
OUT_DIR = "out_direct_multiclass_rare_offline"
MODEL_DIR = Path(OUT_DIR) / "model"

TOPK = 3
AUTO_TH = 0.85
REVIEW_TH = 0.60


# =========================
# 工具
# =========================
def read_csv_smart(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def build_text(row: pd.Series) -> str:
    def norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    company = norm(row.get("company_name", ""))
    tenderee = norm(row.get("tenderee", ""))
    project = norm(row.get("project_name", ""))

    return f"公司：{company}；招标单位：{tenderee}；项目名称：{project}"


def softmax_np(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class TextClsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_col=None, label2id=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_col = label_col
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_text(row)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}

        if self.label_col is not None and self.label2id is not None:
            item["labels"] = torch.tensor(self.label2id[row[self.label_col]], dtype=torch.long)

        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trainer 不是 nn.Module，不能 register_buffer；直接保存为 numpy/torch 都行
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    # 打印运行环境（防止多环境混乱）
    print("PYTHON:", sys.executable)
    print("TRANSFORMERS:", transformers.__version__)
    try:
        import accelerate
        print("ACCELERATE:", accelerate.__version__)
    except Exception as e:
        print("ACCELERATE: NOT INSTALLED -> Trainer 会报错，需要安装 accelerate>=0.26.0")
        raise

    # 强制离线（避免任何联网）
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    set_seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 检查本地模型目录
    assert LOCAL_MODEL_DIR.exists(), f"模型路径不存在：{LOCAL_MODEL_DIR}"
    assert (LOCAL_MODEL_DIR / "config.json").exists(), f"缺少 config.json：{LOCAL_MODEL_DIR}"
    assert (LOCAL_MODEL_DIR / "vocab.txt").exists(), f"缺少 vocab.txt：{LOCAL_MODEL_DIR}"
    assert (LOCAL_MODEL_DIR / "tokenizer_config.json").exists(), f"缺少 tokenizer_config.json：{LOCAL_MODEL_DIR}"
    has_bin = (LOCAL_MODEL_DIR / "pytorch_model.bin").exists()
    has_safe = (LOCAL_MODEL_DIR / "model.safetensors").exists()
    assert (has_bin or has_safe), "缺少权重文件：pytorch_model.bin 或 model.safetensors"

    # 读数据
    df = read_csv_smart(INPUT_CSV)
    for c in TEXT_COLS + [LABEL_COL]:
        if c not in df.columns:
            raise ValueError(f"缺少列：{c}，当前列：{list(df.columns)}")

    # 标注/未标注识别
    df[LABEL_COL] = df[LABEL_COL].astype(str).replace({"nan": "", "None": "", "NaN": ""}).str.strip()
    labeled = df[df[LABEL_COL] != ""].copy()
    unlabeled = df[df[LABEL_COL] == ""].copy()

    print(f"总行数：{len(df)}")
    print(f"已标注：{len(labeled)} | 未标注：{len(unlabeled)}")

    if len(labeled) < 50:
        raise ValueError(f"标注样本太少（{len(labeled)}）。")

    # 稀有类兜底
    vc = labeled[LABEL_COL].value_counts()
    rare_classes = set(vc[vc < MIN_FREQ].index.tolist())
    if rare_classes:
        labeled.loc[labeled[LABEL_COL].isin(rare_classes), LABEL_COL] = RARE_LABEL

    vc2 = labeled[LABEL_COL].value_counts()
    print(f"类别数（稀有类处理后）：{vc2.shape[0]}（原始：{vc.shape[0]}）")
    print(f"被合并进 {RARE_LABEL} 的原始类别数：{len(rare_classes)}")
    print(f"{RARE_LABEL} 样本量：{int(vc2.get(RARE_LABEL, 0))}")

    # label 编码
    labels = sorted(labeled[LABEL_COL].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)

    with open(Path(OUT_DIR) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # 切分 train/val
    stratify = labeled[LABEL_COL] if labeled[LABEL_COL].value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(
        labeled, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
    )

    # class weight（只对训练集出现的类计算）
    y_train = train_df[LABEL_COL].map(label2id).values
    present_classes = np.unique(y_train)
    present_weights = compute_class_weight("balanced", classes=present_classes, y=y_train)

    class_weights = np.ones(num_labels, dtype=np.float32)
    class_weights[present_classes] = present_weights.astype(np.float32)
    class_weights = np.clip(class_weights, 0.3, 10.0)
    print(f"训练集覆盖类别数：{len(present_classes)} / {num_labels}")

    # 从本地加载模型（关键：as_posix + local_files_only）
    local_path = LOCAL_MODEL_DIR.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True
    )

    # Dataset
    train_ds = TextClsDataset(train_df, tokenizer, MAX_LENGTH, label_col=LABEL_COL, label2id=label2id)
    val_ds = TextClsDataset(val_df, tokenizer, MAX_LENGTH, label_col=LABEL_COL, label2id=label2id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments（兼容不同版本）
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    common_kwargs = dict(
        output_dir=OUT_DIR,
        logging_dir=str(Path(OUT_DIR) / "logs"),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16 and torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    if "evaluation_strategy" in ta_params:
        args = TrainingArguments(
            **common_kwargs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
        )
    else:
        args = TrainingArguments(
            **common_kwargs,
            do_train=True,
            do_eval=False,
        )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Train
    trainer.train()

    # Eval report
    val_logits = trainer.predict(val_ds).predictions
    val_probs = softmax_np(val_logits, axis=-1)
    val_pred = np.argmax(val_probs, axis=-1)
    val_true = val_df[LABEL_COL].map(label2id).values

    print("\n=== Validation Report (Top-1) ===")
    print(classification_report(val_true, val_pred, digits=4))

    # Save fine-tuned model
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Predict unlabeled
    if len(unlabeled) > 0:
        unlabeled_ds = TextClsDataset(unlabeled, tokenizer, MAX_LENGTH, label_col=None, label2id=None)
        pred_logits = trainer.predict(unlabeled_ds).predictions
        pred_probs = softmax_np(pred_logits, axis=-1)

        topk_idx = np.argsort(-pred_probs, axis=1)[:, :TOPK]
        topk_prob = np.take_along_axis(pred_probs, topk_idx, axis=1)

        top1_idx = topk_idx[:, 0]
        top1_prob = topk_prob[:, 0]
        top1_label = [id2label[int(i)] for i in top1_idx]

        bucket = np.where(
            top1_prob >= AUTO_TH, "AUTO",
            np.where(top1_prob >= REVIEW_TH, "REVIEW", "MANUAL")
        )

        out = unlabeled.copy()
        out["pred_label"] = top1_label
        out["pred_confidence"] = top1_prob
        out["bucket"] = bucket

        for k in range(TOPK):
            out[f"top{k+1}_label"] = [id2label[int(i)] for i in topk_idx[:, k]]
            out[f"top{k+1}_prob"] = topk_prob[:, k]

        out_path = Path(OUT_DIR) / "pred_unlabeled_with_confidence.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n未标注预测结果已保存：{out_path}")

        # 输出全量填充表
        df_filled = df.copy()
        df_filled.loc[unlabeled.index, LABEL_COL] = out["pred_label"].values
        filled_path = Path(OUT_DIR) / "data_filled_all.csv"
        df_filled.to_csv(filled_path, index=False, encoding="utf-8-sig")
        print(f"全量填充表已保存：{filled_path}")
    else:
        print("没有未标注行可预测。")


if __name__ == "__main__":
    main()
