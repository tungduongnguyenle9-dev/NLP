"""Mini Information Extraction (IE) pipeline & evaluation (Option A — News).

Implements Sections 25, 26, and 27 of LAB 09:
  Raw text -> Fine-tuned NER -> Entity extraction -> Structured JSON
Evaluates entity-level Precision, Recall, and F1 on a 30-sentence News test set
with explicit gold annotations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

DEFAULT_SCRATCH_MODEL_DIR = Path(
    "/Users/user/.gemini/antigravity/brain/d84b4b4c-6174-4ee4-afda-b95415d8f7d8/scratch/ner_model"
)

CONLL_TO_IE_TYPE = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
}
IE_TYPES = ("PERSON", "ORG", "LOCATION")


# 30 News sentences with gold annotations following CoNLL-2003 Option A schema (PERSON, ORG, LOCATION)
NEWS_EVAL_DATA: list[dict[str, Any]] = [
    {
        "text": "Barack Obama visited Paris on Tuesday.",
        "gold": [("PERSON", "Barack Obama"), ("LOCATION", "Paris")],
    },
    {
        "text": "Apple opened a new research office in London.",
        "gold": [("ORG", "Apple"), ("LOCATION", "London")],
    },
    {
        "text": "Satya Nadella met European Union regulators in Brussels.",
        "gold": [("PERSON", "Satya Nadella"), ("ORG", "European Union"), ("LOCATION", "Brussels")],
    },
    {
        "text": "Toyota plans to expand production in Thailand next year.",
        "gold": [("ORG", "Toyota"), ("LOCATION", "Thailand")],
    },
    {
        "text": "Angela Merkel addressed reporters in Berlin after talks with France.",
        "gold": [("PERSON", "Angela Merkel"), ("LOCATION", "Berlin"), ("LOCATION", "France")],
    },
    {
        "text": "Google and Microsoft announced cloud partnerships in Singapore.",
        "gold": [("ORG", "Google"), ("ORG", "Microsoft"), ("LOCATION", "Singapore")],
    },
    {
        "text": "The United Nations held an emergency summit in Geneva.",
        "gold": [("ORG", "United Nations"), ("LOCATION", "Geneva")],
    },
    {
        "text": "Lionel Messi scored twice for Barcelona against Real Madrid.",
        "gold": [("PERSON", "Lionel Messi"), ("ORG", "Barcelona"), ("ORG", "Real Madrid")],
    },
    {
        "text": "The Bank of England kept interest rates unchanged in London.",
        "gold": [("ORG", "Bank of England"), ("LOCATION", "London")],
    },
    {
        "text": "Elon Musk spoke about SpaceX launches from California.",
        "gold": [("PERSON", "Elon Musk"), ("ORG", "SpaceX"), ("LOCATION", "California")],
    },
    {
        "text": "Samsung Electronics reported higher profits in Seoul.",
        "gold": [("ORG", "Samsung Electronics"), ("LOCATION", "Seoul")],
    },
    {
        "text": "Emmanuel Macron welcomed Justin Trudeau to Ottawa.",
        "gold": [("PERSON", "Emmanuel Macron"), ("PERSON", "Justin Trudeau"), ("LOCATION", "Ottawa")],
    },
    {
        "text": "Amazon built two logistics centers near Frankfurt in Germany.",
        "gold": [("ORG", "Amazon"), ("LOCATION", "Frankfurt"), ("LOCATION", "Germany")],
    },
    {
        "text": "The World Health Organization issued travel advice from Geneva.",
        "gold": [("ORG", "World Health Organization"), ("LOCATION", "Geneva")],
    },
    {
        "text": "Tim Cook visited suppliers across Vietnam and Japan.",
        "gold": [("PERSON", "Tim Cook"), ("LOCATION", "Vietnam"), ("LOCATION", "Japan")],
    },
    {
        "text": "Airbus delivered twelve aircraft to Lufthansa in Munich.",
        "gold": [("ORG", "Airbus"), ("ORG", "Lufthansa"), ("LOCATION", "Munich")],
    },
    {
        "text": "Janet Yellen discussed inflation with officials at the Federal Reserve.",
        "gold": [("PERSON", "Janet Yellen"), ("ORG", "Federal Reserve")],
    },
    {
        "text": "Sony partnered with Honda to develop electric vehicles in Tokyo.",
        "gold": [("ORG", "Sony"), ("ORG", "Honda"), ("LOCATION", "Tokyo")],
    },
    {
        "text": "Cristiano Ronaldo joined Manchester United in England.",
        "gold": [("PERSON", "Cristiano Ronaldo"), ("ORG", "Manchester United"), ("LOCATION", "England")],
    },
    {
        "text": "The International Monetary Fund warned Brazil about rising debt.",
        "gold": [("ORG", "International Monetary Fund"), ("LOCATION", "Brazil")],
    },
    {
        "text": "Sundar Pichai attended an AI conference in Zurich.",
        "gold": [("PERSON", "Sundar Pichai"), ("LOCATION", "Zurich")],
    },
    {
        "text": "Boeing signed a contract with Delta Air Lines in Atlanta.",
        "gold": [("ORG", "Boeing"), ("ORG", "Delta Air Lines"), ("LOCATION", "Atlanta")],
    },
    {
        "text": "Narendra Modi met Vladimir Putin during talks in Moscow.",
        "gold": [("PERSON", "Narendra Modi"), ("PERSON", "Vladimir Putin"), ("LOCATION", "Moscow")],
    },
    {
        "text": "Reuters reported heavy rainfall across northern Italy.",
        "gold": [("ORG", "Reuters"), ("LOCATION", "Italy")],
    },
    {
        "text": "Siemens signed an energy agreement with Egypt in Cairo.",
        "gold": [("ORG", "Siemens"), ("LOCATION", "Egypt"), ("LOCATION", "Cairo")],
    },
    {
        "text": "Kylian Mbappe played in Madrid after leaving Paris.",
        "gold": [("PERSON", "Kylian Mbappe"), ("LOCATION", "Madrid"), ("LOCATION", "Paris")],
    },
    {
        "text": "Oil prices fell slightly on Wednesday amid steady global demand.",
        "gold": [],
    },
    {
        "text": "Oracle expanded its data center network in Sydney and Melbourne.",
        "gold": [("ORG", "Oracle"), ("LOCATION", "Sydney"), ("LOCATION", "Melbourne")],
    },
    {
        "text": "Christine Lagarde spoke at the European Central Bank in Frankfurt.",
        "gold": [("PERSON", "Christine Lagarde"), ("ORG", "European Central Bank"), ("LOCATION", "Frankfurt")],
    },
    {
        "text": "Nvidia announced new chips designed in Santa Clara.",
        "gold": [("ORG", "Nvidia"), ("LOCATION", "Santa Clara")],
    },
]


class NewsInformationExtractor:
    """Extract structured PERSON / ORG / LOCATION entities from raw text."""

    def __init__(self, model_dir: str | Path = DEFAULT_SCRATCH_MODEL_DIR):
        model_path = Path(model_dir)
        if model_path.exists():
            source = str(model_path)
        else:
            raise FileNotFoundError(
                f"Fine-tuned NER model not found at {model_path}. "
                "Please execute ner_finetuning.ipynb first or pass --model-dir."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForTokenClassification.from_pretrained(source)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def _extract_word_spans_and_tags(self, text: str) -> tuple[list[tuple[int, int]], list[str]]:
        """Run tokenization, NER model forward pass, and Strategy-A word alignment."""
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            pred_ids = logits.argmax(dim=-1).tolist()

        word_spans: dict[int, list[int]] = {}
        word_tags: dict[int, str] = {}
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            start_c, end_c = offsets[pos]
            if wid not in word_spans:
                word_spans[wid] = [start_c, end_c]
                word_tags[wid] = self.id2label[pred_ids[pos]]
            else:
                word_spans[wid][1] = end_c

        sorted_wids = sorted(word_spans.keys())
        spans = [(word_spans[w][0], word_spans[w][1]) for w in sorted_wids]
        tags = [word_tags[w] for w in sorted_wids]
        return spans, tags

    def predict_word_bio(self, text: str) -> tuple[list[str], list[str]]:
        """Tokenize raw text, run NER model, and align predictions back to words."""
        spans, tags = self._extract_word_spans_and_tags(text)
        words = [text[s:e] for s, e in spans]
        return words, tags

    def extract_entities(self, text: str) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
        """Decode BIO tags into ordered entity tuples and structured Option-A JSON."""
        spans, tags = self._extract_word_spans_and_tags(text)

        entities: list[tuple[str, str]] = []
        cur_type: str | None = None
        cur_start: int | None = None
        cur_end: int | None = None

        def flush_current():
            nonlocal cur_type, cur_start, cur_end
            if cur_type is not None and cur_start is not None and cur_end is not None:
                ie_type = CONLL_TO_IE_TYPE.get(cur_type)
                if ie_type is not None:
                    entities.append((ie_type, text[cur_start:cur_end]))
            cur_type, cur_start, cur_end = None, None, None

        for (w_start, w_end), tag in zip(spans, tags):
            if tag == "O":
                flush_current()
            elif tag.startswith("B-"):
                flush_current()
                cur_type = tag[2:]
                cur_start, cur_end = w_start, w_end
            elif tag.startswith("I-"):
                etype = tag[2:]
                if cur_type == etype:
                    cur_end = w_end
                else:
                    # Recover gracefully if I-X starts without matching B-X
                    flush_current()
                    cur_type = etype
                    cur_start, cur_end = w_start, w_end

        flush_current()

        structured: dict[str, list[str]] = {t: [] for t in IE_TYPES}
        for etype, etext in entities:
            if etext not in structured[etype]:
                structured[etype].append(etext)

        return entities, structured


def compute_entity_metrics(
    gold_all: list[list[tuple[str, str]]],
    pred_all: list[list[tuple[str, str]]],
) -> dict[str, Any]:
    """Compute strict entity-level Precision, Recall, and F1 overall and per IE type."""
    tp_total = fp_total = fn_total = 0
    per_type_counts: dict[str, dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in IE_TYPES}

    for g_list, p_list in zip(gold_all, pred_all):
        g_cnt = Counter(g_list)
        p_cnt = Counter(p_list)
        matched = g_cnt & p_cnt
        fp_cnt = p_cnt - g_cnt
        fn_cnt = g_cnt - p_cnt

        tp_total += sum(matched.values())
        fp_total += sum(fp_cnt.values())
        fn_total += sum(fn_cnt.values())

        for (etype, _), c in matched.items():
            if etype in per_type_counts:
                per_type_counts[etype]["tp"] += c
        for (etype, _), c in fp_cnt.items():
            if etype in per_type_counts:
                per_type_counts[etype]["fp"] += c
        for (etype, _), c in fn_cnt.items():
            if etype in per_type_counts:
                per_type_counts[etype]["fn"] += c

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        "overall": prf(tp_total, fp_total, fn_total),
        "per_type": {t: prf(c["tp"], c["fp"], c["fn"]) for t, c in per_type_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Run Option A (News) Information Extraction & 30-sentence Eval.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_SCRATCH_MODEL_DIR),
        help="Path to fine-tuned NER model checkpoint directory.",
    )
    args = parser.parse_args()

    extractor = NewsInformationExtractor(args.model_dir)

    print("=== Sample IE Structured Output (Section 25) ===")
    for sample_idx in [0, 2, 8]:
        text = NEWS_EVAL_DATA[sample_idx]["text"]
        _, structured = extractor.extract_entities(text)
        print(f"Input : {text}")
        print(f"Output: {json.dumps(structured, ensure_ascii=False)}")

    print(f"\n=== Option A (News) IE Evaluation on {len(NEWS_EVAL_DATA)} Gold Sentences (Section 27) ===")
    gold_all = []
    pred_all = []
    mismatches = []

    for idx, item in enumerate(NEWS_EVAL_DATA):
        g_ents = [tuple(x) for x in item["gold"]]
        p_ents, _ = extractor.extract_entities(item["text"])
        gold_all.append(g_ents)
        pred_all.append(p_ents)
        if Counter(g_ents) != Counter(p_ents):
            mismatches.append((idx, item["text"], g_ents, p_ents))

    metrics = compute_entity_metrics(gold_all, pred_all)
    ov = metrics["overall"]
    print(
        f"Overall Entity-Level -> Precision: {ov['precision']:.4f} | "
        f"Recall: {ov['recall']:.4f} | F1: {ov['f1']:.4f} "
        f"(TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']})"
    )
    print("\nPer-Type Breakdown:")
    for t in IE_TYPES:
        m = metrics["per_type"][t]
        print(
            f"  {t:<8} | P={m['precision']:.4f} | R={m['recall']:.4f} | "
            f"F1={m['f1']:.4f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})"
        )

    if mismatches:
        print(f"\nSentences with mismatches ({len(mismatches)}/{len(NEWS_EVAL_DATA)}):")
        for idx, txt, g_e, p_e in mismatches:
            print(f"  [{idx:02d}] {txt}")
            print(f"       Gold: {g_e}")
            print(f"       Pred: {p_e}")
    else:
        print("\nAll 30 sentences matched gold entities perfectly!")


if __name__ == "__main__":
    main()
