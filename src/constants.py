from __future__ import annotations

SUPPORTED_METHODS = [
    "full_ft",
    "lora",
    "topheavy_lora",
    "bitfit",
    "gradaware_lora",
]

SUPPORTED_TASKS = ["sst2", "mrpc", "cola", "qnli", "rte"]

SUPPORTED_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

TASK_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "qnli": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
}

DEFAULT_TRAIN_SUBSET = 500
DEFAULT_MAX_LENGTH = 256
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
