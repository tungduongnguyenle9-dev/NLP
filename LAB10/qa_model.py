"""Core modules for LAB 10 — Extractive Question Answering.

Includes:
- SQuAD text normalization, Exact Match (EM), and Token-level F1 metrics.
- Offset mapping alignment between character spans and subword token spans.
- Pretrained Transformer backbone with Start/End linear classification heads.
- Valid span decoding (i <= j) across sliding windows.
- Simple Keyword-Overlap Baseline for comparison against the Transformer.
"""

from __future__ import annotations

import collections
import inspect
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "over", "after", "and", "or", "but", "if", "as",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "hers", "we", "our", "you", "your",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "can", "could", "would", "should", "may", "might", "will", "shall",
    "did", "not", "no", "so", "than", "too", "very", "such", "there",
}


# ---------------------------------------------------------------------------
# 1. Normalization & Evaluation Metrics (Exact Match & Token-level F1)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_tokens(text: str) -> List[str]:
    """Return normalized whitespace-delimited tokens for QA evaluation."""
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized prediction exactly matches ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between a single prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: Sequence[str]
) -> float:
    """Compute the maximum score over all valid reference answers for a sample."""
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate_qa_predictions(
    predictions: Dict[str, str],
    references: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    """Evaluate predictions dict {id: pred_text} against {id: [gold_texts]}.

    Returns percentages in [0, 100] under keys 'exact_match' and 'f1'.
    """
    total = 0
    em_sum = 0.0
    f1_sum = 0.0
    for qid, gold_list in references.items():
        total += 1
        pred = predictions.get(qid, "")
        em_sum += metric_max_over_ground_truths(compute_exact_match, pred, gold_list)
        f1_sum += metric_max_over_ground_truths(compute_f1, pred, gold_list)

    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "total": 0}
    return {
        "exact_match": 100.0 * em_sum / total,
        "f1": 100.0 * f1_sum / total,
        "total": total,
    }


# ---------------------------------------------------------------------------
# 2. Offset Mapping & Character-to-Token Span Alignment
# ---------------------------------------------------------------------------

def align_answer_to_tokens(
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    answer_start: int,
    answer_end: int,
    cls_index: int = 0,
) -> Dict[str, Any]:
    """Map character span [answer_start, answer_end) into token indices (start, end).

    Handles the 4 required cases:
    - Answer starts in the middle of a token.
    - Answer ends in the middle of a token.
    - Answer is truncated by window boundary (partially inside window).
    - Answer lies completely outside the current window.
    """
    if answer_end <= answer_start:
        return {
            "start_token": cls_index,
            "end_token": cls_index,
            "status": "empty_answer",
            "starts_mid_token": False,
            "ends_mid_token": False,
        }

    # Find contiguous context token bounds (sequence_ids == 1).
    context_start = 0
    while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
        context_start += 1

    if context_start >= len(sequence_ids):
        return {
            "start_token": cls_index,
            "end_token": cls_index,
            "status": "no_context_tokens",
            "starts_mid_token": False,
            "ends_mid_token": False,
        }

    context_end = len(sequence_ids) - 1
    while context_end >= 0 and sequence_ids[context_end] != 1:
        context_end -= 1

    window_char_start = offsets[context_start][0]
    window_char_end = offsets[context_end][1]

    # Case 4: Answer is completely outside the current sliding window.
    if answer_end <= window_char_start or answer_start >= window_char_end:
        return {
            "start_token": cls_index,
            "end_token": cls_index,
            "status": "out_of_window",
            "starts_mid_token": False,
            "ends_mid_token": False,
        }

    # Case 3: Answer crosses the left or right boundary of this window (truncated).
    if answer_start < window_char_start or answer_end > window_char_end:
        return {
            "start_token": cls_index,
            "end_token": cls_index,
            "status": "truncated",
            "starts_mid_token": False,
            "ends_mid_token": False,
        }

    # Answer fits completely inside this window.
    start_token: Optional[int] = None
    end_token: Optional[int] = None

    for token_idx in range(context_start, context_end + 1):
        tok_start, tok_end = offsets[token_idx]
        if tok_start == tok_end:
            continue
        if tok_start <= answer_start < tok_end and start_token is None:
            start_token = token_idx
        if tok_start < answer_end <= tok_end:
            end_token = token_idx

    # Fallback if answer_start or answer_end fell on stripped whitespace between tokens.
    if start_token is None:
        idx = context_start
        while idx <= context_end and offsets[idx][0] <= answer_start:
            idx += 1
        start_token = max(context_start, idx - 1)
        if offsets[start_token][1] <= answer_start and start_token + 1 <= context_end:
            start_token += 1

    if end_token is None:
        idx = context_end
        while idx >= context_start and offsets[idx][1] >= answer_end:
            idx -= 1
        end_token = min(context_end, idx + 1)
        if offsets[end_token][0] >= answer_end and end_token - 1 >= context_start:
            end_token -= 1

    if start_token > end_token:
        start_token = end_token

    tok_s0, tok_s1 = offsets[start_token]
    tok_e0, tok_e1 = offsets[end_token]
    starts_mid_token = tok_s0 < answer_start < tok_s1
    ends_mid_token = tok_e0 < answer_end < tok_e1

    return {
        "start_token": int(start_token),
        "end_token": int(end_token),
        "status": "aligned",
        "starts_mid_token": bool(starts_mid_token),
        "ends_mid_token": bool(ends_mid_token),
    }


def prepare_train_features(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_length: int = 384,
    stride: int = 128,
) -> Dict[str, List[Any]]:
    """Tokenize QA pairs with sliding windows and attach start_positions/end_positions."""
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []

    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(cls_token_id) if cls_token_id in input_ids else 0
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end = answer_start + len(answer_text)

        alignment = align_answer_to_tokens(
            offsets=offsets,
            sequence_ids=sequence_ids,
            answer_start=answer_start,
            answer_end=answer_end,
            cls_index=cls_index,
        )
        start_positions.append(alignment["start_token"])
        end_positions.append(alignment["end_token"])

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_eval_features(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_length: int = 384,
    stride: int = 128,
) -> Dict[str, List[Any]]:
    """Tokenize evaluation QA pairs keeping context-only offsets and example_id."""
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        # Keep offsets only for context tokens (sequence_id == 1); mark others None.
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

    return tokenized


# ---------------------------------------------------------------------------
# 3. Extractive QA Transformer Model (Custom Start/End Prediction Head)
# ---------------------------------------------------------------------------

class ExtractiveQAModel(nn.Module):
    """Pretrained Transformer backbone with linear Start/End span prediction head.

    Architecture:
        (Question + Context) -> Backbone -> H [B, T, D] -> Linear(D, 2) -> start_logits, end_logits
    Loss:
        L = (CE(start_logits, start_positions) + CE(end_logits, end_positions)) / 2
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self._accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(self.backbone.forward).parameters
        )
        hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "dim", 768))
        self.dropout = nn.Dropout(dropout_prob)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._accepts_token_type_ids and token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, T, H]
        logits = self.qa_outputs(sequence_output)  # [B, T, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [B, T]
        end_logits = end_logits.squeeze(-1)      # [B, T]

        result: Dict[str, torch.Tensor] = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_pos = start_positions.clamp(0, ignored_index)
            end_pos = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_pos)
            end_loss = loss_fct(end_logits, end_pos)
            total_loss = (start_loss + end_loss) / 2.0

            result["loss"] = total_loss
            result["start_loss"] = start_loss
            result["end_loss"] = end_loss

        return result

    def save_model(self, save_dir: str) -> None:
        """Save model metadata and state_dict without using HF pipeline abstractions."""
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "dropout_prob": self.dropout.p,
        }
        with open(os.path.join(save_dir, "qa_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.pt"))

    @classmethod
    def load_model(cls, save_dir: str, device: Optional[torch.device] = None) -> "ExtractiveQAModel":
        """Restore a saved ExtractiveQAModel from disk."""
        with open(os.path.join(save_dir, "qa_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(
            model_name=meta["model_name"],
            dropout_prob=meta.get("dropout_prob", 0.1),
        )
        map_location = device if device is not None else torch.device("cpu")
        state = torch.load(
            os.path.join(save_dir, "pytorch_model.pt"),
            map_location=map_location,
        )
        model.load_state_dict(state)
        if device is not None:
            model.to(device)
        return model


# ---------------------------------------------------------------------------
# 4. Span Decoding / Post-Processing (Constraint i <= j)
# ---------------------------------------------------------------------------

def decode_qa_predictions(
    examples: Sequence[Dict[str, Any]],
    features: Sequence[Dict[str, Any]],
    raw_start_logits,
    raw_end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Select the highest-scoring valid context span (i <= j) per example."""
    features_per_example: Dict[str, List[int]] = collections.defaultdict(list)
    for feat_idx, feat in enumerate(features):
        features_per_example[feat["example_id"]].append(feat_idx)

    predictions: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}

    for example in examples:
        ex_id = example["id"]
        context = example["context"]
        feature_indices = features_per_example.get(ex_id, [])

        best_candidate: Optional[Dict[str, Any]] = None

        for feat_idx in feature_indices:
            start_logits = raw_start_logits[feat_idx]
            end_logits = raw_end_logits[feat_idx]
            offset_mapping = features[feat_idx]["offset_mapping"]

            start_indexes = sorted(
                range(len(start_logits)),
                key=lambda idx: float(start_logits[idx]),
                reverse=True,
            )[:n_best_size]
            end_indexes = sorted(
                range(len(end_logits)),
                key=lambda idx: float(end_logits[idx]),
                reverse=True,
            )[:n_best_size]

            for s_idx in start_indexes:
                if s_idx >= len(offset_mapping) or offset_mapping[s_idx] is None:
                    continue
                for e_idx in end_indexes:
                    if e_idx >= len(offset_mapping) or offset_mapping[e_idx] is None:
                        continue
                    if e_idx < s_idx or (e_idx - s_idx + 1) > max_answer_length:
                        continue

                    char_start = offset_mapping[s_idx][0]
                    char_end = offset_mapping[e_idx][1]
                    if char_end <= char_start:
                        continue

                    score = float(start_logits[s_idx]) + float(end_logits[e_idx])
                    if best_candidate is None or score > best_candidate["score"]:
                        best_candidate = {
                            "score": score,
                            "text": context[char_start:char_end],
                            "char_start": int(char_start),
                            "char_end": int(char_end),
                            "start_token": int(s_idx),
                            "end_token": int(e_idx),
                        }

        if best_candidate is None:
            predictions[ex_id] = ""
            details[ex_id] = {
                "score": 0.0,
                "text": "",
                "char_start": -1,
                "char_end": -1,
                "start_token": 0,
                "end_token": 0,
            }
        else:
            predictions[ex_id] = best_candidate["text"]
            details[ex_id] = best_candidate

    return predictions, details


def answer_question_single(
    model: ExtractiveQAModel,
    tokenizer,
    question: str,
    context: str,
    device: Optional[torch.device] = None,
    max_length: int = 384,
    stride: int = 128,
    max_answer_length: int = 30,
) -> Dict[str, Any]:
    """Run extractive QA inference on a single (question, context) pair."""
    if device is None:
        device = next(model.parameters()).device

    batch_dict = {
        "id": ["single_0"],
        "question": [question],
        "context": [context],
    }
    eval_features = prepare_eval_features(
        batch_dict, tokenizer=tokenizer, max_length=max_length, stride=stride
    )

    num_windows = len(eval_features["input_ids"])
    feature_list = [
        {
            "example_id": eval_features["example_id"][i],
            "offset_mapping": eval_features["offset_mapping"][i],
        }
        for i in range(num_windows)
    ]

    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(eval_features["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(eval_features["attention_mask"], dtype=torch.long, device=device)
        token_type_ids = None
        if "token_type_ids" in eval_features:
            token_type_ids = torch.tensor(
                eval_features["token_type_ids"], dtype=torch.long, device=device
            )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        start_logits = outputs["start_logits"].cpu().tolist()
        end_logits = outputs["end_logits"].cpu().tolist()

    examples = [{"id": "single_0", "question": question, "context": context}]
    _, details = decode_qa_predictions(
        examples=examples,
        features=feature_list,
        raw_start_logits=start_logits,
        raw_end_logits=end_logits,
        max_answer_length=max_answer_length,
    )
    info = details["single_0"]
    info["num_windows"] = num_windows
    return info


# ---------------------------------------------------------------------------
# 5. Simple Heuristic Baseline (Keyword Overlap -> Sentence -> Span)
# ---------------------------------------------------------------------------

class KeywordOverlapBaseline:
    """Simple non-neural extractive QA baseline.

    Steps:
    1. Split context into sentences.
    2. Select the sentence with highest non-stopword overlap with the question.
    3. Extract a short candidate span inside that sentence that does not duplicate
       question words (preferring capitalized entities, numbers, or short noun chunks).
    """

    @staticmethod
    def _content_words(text: str) -> List[str]:
        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        return [w for w in words if w not in STOPWORDS and len(w) > 1]

    @staticmethod
    def _split_sentences(context: str) -> List[str]:
        sents = re.split(r"(?<=[.!?])\s+", context.strip())
        sents = [s.strip() for s in sents if s.strip()]
        return sents if sents else [context.strip()]

    def predict(self, question: str, context: str) -> str:
        """Predict an extractive span using keyword overlap heuristics."""
        if not context.strip():
            return ""
        q_keywords = set(self._content_words(question))
        sentences = self._split_sentences(context)

        # Pick sentence with maximum keyword overlap (tie-break by overlap ratio).
        best_sent = sentences[0]
        best_score = (-1.0, -1.0)
        for sent in sentences:
            s_words = self._content_words(sent)
            if not s_words:
                continue
            overlap = len(q_keywords.intersection(s_words))
            ratio = overlap / (len(s_words) ** 0.5)
            score = (float(overlap), ratio)
            if score > best_score:
                best_score = score
                best_sent = sent

        # Extract candidate span from best_sent: contiguous tokens not in question keywords.
        token_matches = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*", best_sent))
        if not token_matches:
            return best_sent

        chunks: List[Tuple[int, int, str, float]] = []
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None
        cur_tokens: List[str] = []

        def flush_chunk() -> None:
            nonlocal cur_start, cur_end, cur_tokens
            if cur_start is not None and cur_end is not None and cur_tokens:
                span_text = best_sent[cur_start:cur_end]
                # Heuristic priority: digits/capitalized phrases, moderate length (1-4 words).
                has_digit = any(any(ch.isdigit() for ch in t) for t in cur_tokens)
                has_cap = any(t[0].isupper() for t in cur_tokens)
                length_penalty = abs(len(cur_tokens) - 2.0) * 0.25
                chunk_score = (2.0 if has_digit else 0.0) + (1.5 if has_cap else 0.0) - length_penalty
                chunks.append((cur_start, cur_end, span_text, chunk_score))
            cur_start = None
            cur_end = None
            cur_tokens = []

        for m in token_matches:
            tok = m.group(0)
            tok_low = tok.lower()
            is_novel_content = tok_low not in q_keywords and tok_low not in STOPWORDS
            if is_novel_content and len(cur_tokens) < 4:
                if cur_start is None:
                    cur_start = m.start()
                cur_end = m.end()
                cur_tokens.append(tok)
            else:
                flush_chunk()
        flush_chunk()

        if chunks:
            best_chunk = max(chunks, key=lambda item: item[3])
            return best_chunk[2]

        # Fallback: first 3 tokens of the selected sentence.
        fallback_end = token_matches[min(2, len(token_matches) - 1)].end()
        return best_sent[token_matches[0].start():fallback_end]

    def predict_batch(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        """Predict spans for a collection of SQuAD-style examples."""
        return {ex["id"]: self.predict(ex["question"], ex["context"]) for ex in examples}


# ---------------------------------------------------------------------------
# 6. Self-test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Running qa_model.py self-tests ===")

    # 1. Metric tests
    assert compute_exact_match("The Paris", "paris") == 1.0
    f1_demo = compute_f1("France", "Paris France")
    assert abs(f1_demo - (2.0 / 3.0)) < 1e-6, f"Unexpected F1: {f1_demo}"
    print(f"[OK] Metrics: EM('The Paris', 'paris') = 1.0, F1('France', 'Paris France') = {f1_demo:.4f}")

    # 2. Baseline test
    baseline = KeywordOverlapBaseline()
    b_pred = baseline.predict("What is the capital of France?", "The capital of France is Paris.")
    assert "Paris" in b_pred, f"Unexpected baseline prediction: {b_pred}"
    print(f"[OK] Baseline prediction: '{b_pred}'")

    # 3. Tokenizer & Alignment test
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sample = {
        "id": ["demo_1"],
        "question": ["What is the capital of France?"],
        "context": ["The capital of France is Paris."],
        "answers": [{"text": ["Paris"], "answer_start": [25]}],
    }
    train_feat = prepare_train_features(sample, tok, max_length=32, stride=8)
    s_pos = train_feat["start_positions"][0]
    e_pos = train_feat["end_positions"][0]
    decoded = tok.decode(train_feat["input_ids"][0][s_pos : e_pos + 1])
    assert decoded.strip() == "paris", f"Unexpected aligned span: {decoded}"
    print(f"[OK] Alignment: start={s_pos}, end={e_pos}, decoded='{decoded}'")

    # 4. Model forward + loss shape test
    model = ExtractiveQAModel("distilbert-base-uncased")
    input_ids = torch.tensor(train_feat["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(train_feat["attention_mask"], dtype=torch.long)
    start_pos = torch.tensor(train_feat["start_positions"], dtype=torch.long)
    end_pos = torch.tensor(train_feat["end_positions"], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_pos,
        end_positions=end_pos,
    )
    assert out["start_logits"].shape == input_ids.shape
    assert out["end_logits"].shape == input_ids.shape
    assert out["loss"].ndim == 0 and out["loss"].item() > 0
    print(
        f"[OK] Model forward: logits shape={tuple(out['start_logits'].shape)}, "
        f"loss={out['loss'].item():.4f}"
    )
    print("=== All qa_model.py self-tests PASSED ===")
