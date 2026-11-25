"""Baseline NER model using token-level features and Logistic Regression on CoNLL-2003.

Implements Section 13 (Baseline) of LAB 09 so we have a simple feature-based
baseline evaluated on the exact same CoNLL-2003 test split (3,453 sentences)
as the fine-tuned Transformer model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1,
    precision_score as sklearn_precision,
    recall_score as sklearn_recall,
)

LABEL_LIST = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def load_conll2003():
    """Load CoNLL-2003 dataset from Hugging Face Hub."""
    try:
        return load_dataset("lhoestq/conll2003")
    except Exception:
        return load_dataset("eriktks/conll2003", trust_remote_code=True)


def word_shape(word: str) -> str:
    """Map characters to coarse shape markers (X=upper, x=lower, d=digit)."""
    chars = []
    for ch in word:
        if ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)
    # Collapse repeated shape letters slightly for compact generalization
    return re.sub(r"(X{2,}|x{2,}|d{2,})", lambda m: m.group(0)[0] * 2, "".join(chars))


def extract_token_features(tokens: list[str], idx: int) -> dict[str, Any]:
    """Extract lexical, orthographic, and local context features for token `idx`."""
    word = tokens[idx]
    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.suffix3": word[-3:].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.prefix2": word[:2].lower(),
        "word.prefix3": word[:3].lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_digit": any(ch.isdigit() for ch in word),
        "word.has_hyphen": "-" in word,
        "word.is_punct": all(not ch.isalnum() for ch in word),
        "word.shape": word_shape(word),
        "word.len": min(len(word), 15),
    }

    if idx > 0:
        prev_word = tokens[idx - 1]
        features.update(
            {
                "-1:word.lower": prev_word.lower(),
                "-1:word.istitle": prev_word.istitle(),
                "-1:word.isupper": prev_word.isupper(),
                "-1:word.isdigit": prev_word.isdigit(),
                "-1:word.shape": word_shape(prev_word),
            }
        )
    else:
        features["BOS"] = True

    if idx < len(tokens) - 1:
        next_word = tokens[idx + 1]
        features.update(
            {
                "+1:word.lower": next_word.lower(),
                "+1:word.istitle": next_word.istitle(),
                "+1:word.isupper": next_word.isupper(),
                "+1:word.isdigit": next_word.isdigit(),
                "+1:word.shape": word_shape(next_word),
            }
        )
    else:
        features["EOS"] = True

    return features


def prepare_split(split_data) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    """Flatten sentence tokens into feature dicts, string BIO labels, and sentence lengths."""
    X_dicts: list[dict[str, Any]] = []
    y_labels: list[str] = []
    sent_lengths: list[int] = []

    for row in split_data:
        tokens = row["tokens"]
        tag_ids = row["ner_tags"]
        sent_lengths.append(len(tokens))
        for i in range(len(tokens)):
            X_dicts.append(extract_token_features(tokens, i))
            y_labels.append(LABEL_LIST[tag_ids[i]])

    return X_dicts, y_labels, sent_lengths


def unflatten_by_sentence(flat_labels: list[str], sent_lengths: list[int]) -> list[list[str]]:
    """Group flat token-level BIO labels back into per-sentence sequences."""
    sequences: list[list[str]] = []
    offset = 0
    for length in sent_lengths:
        sequences.append(flat_labels[offset : offset + length])
        offset += length
    return sequences


def train_and_evaluate_baseline(max_iter: int = 200) -> dict[str, Any]:
    """Train Logistic Regression baseline on CoNLL-2003 train and evaluate on test."""
    t0 = time.time()
    dataset = load_conll2003()

    print(
        f"Loaded CoNLL-2003 — train: {len(dataset['train'])}, "
        f"validation: {len(dataset['validation'])}, test: {len(dataset['test'])}"
    )

    X_train_dicts, y_train_flat, _ = prepare_split(dataset["train"])
    X_test_dicts, y_test_flat, test_lengths = prepare_split(dataset["test"])

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_test = vectorizer.transform(X_test_dicts)
    print(f"Feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    clf = LogisticRegression(
        max_iter=max_iter,
        C=1.0,
        solver="lbfgs",
        tol=1e-3,
        random_state=42,
    )
    clf.fit(X_train, y_train_flat)

    y_pred_flat = list(clf.predict(X_test))

    y_true_seqs = unflatten_by_sentence(y_test_flat, test_lengths)
    y_pred_seqs = unflatten_by_sentence(y_pred_flat, test_lengths)

    entity_p = float(seqeval_precision(y_true_seqs, y_pred_seqs))
    entity_r = float(seqeval_recall(y_true_seqs, y_pred_seqs))
    entity_f1 = float(seqeval_f1(y_true_seqs, y_pred_seqs))
    entity_report = seqeval_classification_report(y_true_seqs, y_pred_seqs, digits=4)

    non_o_labels = [lbl for lbl in LABEL_LIST if lbl != "O"]
    token_acc = float(accuracy_score(y_test_flat, y_pred_flat))
    token_p_non_o = float(
        sklearn_precision(y_test_flat, y_pred_flat, labels=non_o_labels, average="micro", zero_division=0)
    )
    token_r_non_o = float(
        sklearn_recall(y_test_flat, y_pred_flat, labels=non_o_labels, average="micro", zero_division=0)
    )
    token_f1_non_o = float(
        sklearn_f1(y_test_flat, y_pred_flat, labels=non_o_labels, average="micro", zero_division=0)
    )

    elapsed = time.time() - t0
    results = {
        "model": "Baseline (Token Features + Logistic Regression)",
        "test_sentences": len(dataset["test"]),
        "test_tokens": len(y_test_flat),
        "num_features": X_train.shape[1],
        "entity_precision": round(entity_p, 4),
        "entity_recall": round(entity_r, 4),
        "entity_f1": round(entity_f1, 4),
        "token_accuracy": round(token_acc, 4),
        "token_precision_non_o": round(token_p_non_o, 4),
        "token_recall_non_o": round(token_r_non_o, 4),
        "token_f1_non_o": round(token_f1_non_o, 4),
        "elapsed_sec": round(elapsed, 2),
        "entity_report": entity_report,
    }

    print("\n=== Baseline NER Evaluation (CoNLL-2003 Test Set) ===")
    print(f"Entity-level Precision : {entity_p:.4f}")
    print(f"Entity-level Recall    : {entity_r:.4f}")
    print(f"Entity-level F1        : {entity_f1:.4f}")
    print(f"Token-level Accuracy   : {token_acc:.4f}  (note: inflated by 'O' tags)")
    print(f"Token-level F1 (non-O) : {token_f1_non_o:.4f}")
    print("\nPer-Entity Classification Report (seqeval):")
    print(entity_report)
    print(f"Completed in {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Logistic Regression Baseline NER on CoNLL-2003.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations for LogisticRegression (default: 200).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary JSON.",
    )
    args = parser.parse_args()

    results = train_and_evaluate_baseline(max_iter=args.max_iter)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
