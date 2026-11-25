"""
document_qa.py — Mini Document Extractive QA System (Section 27)
LAB 10: Extractive Question Answering

Uses the fine-tuned ExtractiveQAModel (from qa_model.py) directly on a given
context/document (no RAG / retrieval required in LAB 10).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

# Ensure local import works whether invoked from LAB10/ or parent directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from qa_model import ExtractiveQAModel, answer_question_single  # noqa: E402


DEFAULT_MODEL_DIR = Path.home() / ".cache" / "lab10_qa_model"

SAMPLE_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "wikipedia": {
        "title": "Wikipedia excerpt — Transformer (deep learning)",
        "context": (
            "A transformer is a deep learning architecture developed by researchers at Google "
            "and based on the multi-head attention mechanism, proposed in a 2017 paper titled "
            "'Attention Is All You Need'. Text is converted to numerical representations called "
            "tokens, and each token is converted into a vector via looking up from a word "
            "embedding table. Unlike recurrent neural networks, transformers process all tokens "
            "in parallel, which allows much faster training on large datasets. Pretrained "
            "transformers such as BERT and DistilBERT can be fine-tuned for sequence "
            "classification, named entity recognition, and extractive question answering."
        ),
        "sample_questions": [
            "Who developed the transformer architecture?",
            "In what year was the paper 'Attention Is All You Need' proposed?",
            "Why do transformers allow faster training than recurrent neural networks?",
        ],
    },
    "academic": {
        "title": "Academic note — SQuAD Dataset",
        "context": (
            "The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset "
            "consisting of questions posed by crowdworkers on a set of Wikipedia articles, where "
            "the answer to every question is a segment of text, or span, from the corresponding "
            "reading passage. SQuAD v1.1 contains over 100,000 question-answer pairs on more "
            "than 500 articles. Models are evaluated using two main metrics: Exact Match (EM), "
            "which checks whether the normalized predicted span matches the gold answer exactly, "
            "and token-level F1, which measures the overlap between predicted and gold tokens."
        ),
        "sample_questions": [
            "How many question-answer pairs does SQuAD v1.1 contain?",
            "What are the two main metrics used to evaluate models on SQuAD?",
        ],
    },
    "faq": {
        "title": "Course FAQ — NLP Lab Submission",
        "context": (
            "Students must submit their lab assignments before 23:59 on the deadline date. "
            "Each submission folder should include the Python source files, executed Jupyter "
            "notebooks, evaluation charts, and written analysis reports in Markdown format. "
            "Late submissions are penalized by 10 percent per day up to three days. For "
            "technical questions about datasets or PyTorch environments, students can contact "
            "the teaching assistant via the course forum or during Friday office hours."
        ),
        "sample_questions": [
            "What is the penalty for late submissions?",
            "When can students meet the teaching assistant in person?",
        ],
    },
}


class DocumentQASystem:
    """Mini Extractive QA engine for answering questions given a context document."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        max_length: int = 384,
        stride: int = 128,
        max_answer_length: int = 30,
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.stride = stride
        self.max_answer_length = max_answer_length

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        checkpoint_path = self.model_dir / "pytorch_model.pt"
        meta_path = self.model_dir / "qa_meta.json"
        if not checkpoint_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found in {self.model_dir}.\n"
                "Please run qa_finetuning.ipynb first to train and save the QA model."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = ExtractiveQAModel.load_model(str(self.model_dir), device=self.device)
        self.base_model_name = self.model.model_name
        self.model.eval()

    @torch.no_grad()
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """Run sliding-window tokenization and span decoding to extract the best answer span."""
        result = answer_question_single(
            model=self.model,
            tokenizer=self.tokenizer,
            question=question,
            context=context,
            device=self.device,
            max_length=self.max_length,
            stride=self.stride,
            max_answer_length=self.max_answer_length,
        )
        result["question"] = question
        return result


def highlight_span(context: str, char_start: int, char_end: int, margin: int = 80) -> str:
    """Return a short snippet of context with the extracted answer span highlighted."""
    if char_end <= char_start:
        return context[:160]
    left = max(0, char_start - margin)
    right = min(len(context), char_end + margin)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(context) else ""
    return (
        f"{prefix}{context[left:char_start]}"
        f"[[{context[char_start:char_end]}]]"
        f"{context[char_end:right]}{suffix}"
    )


def run_demo(qa: DocumentQASystem) -> None:
    print("=" * 72)
    print("LAB 10 — Mini Document QA Demo (Fine-tuned Extractive QA Model)")
    print("=" * 72)
    for key, doc in SAMPLE_DOCUMENTS.items():
        print(f"\n[{key.upper()}] {doc['title']}")
        print("-" * 72)
        for q in doc["sample_questions"]:
            res = qa.answer(q, doc["context"])
            print(f"Q: {q}")
            print(
                f"A: {res['text']}  "
                f"(char_span=({res['char_start']}, {res['char_end']}), score={res['score']:.2f})"
            )
            print(f"   Snippet: {highlight_span(doc['context'], res['char_start'], res['char_end'])}")


def run_interactive(qa: DocumentQASystem) -> None:
    print("Interactive Document QA Mode (type 'quit' to exit)")
    print("Choose a built-in sample (wikipedia / academic / faq) or type 'custom':")
    choice = input("> ").strip().lower()
    if choice in SAMPLE_DOCUMENTS:
        context = SAMPLE_DOCUMENTS[choice]["context"]
        print(f"\nLoaded document: {SAMPLE_DOCUMENTS[choice]['title']}\n{context}\n")
    else:
        print("Paste your context paragraph (single line):")
        context = input("> ").strip()

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        res = qa.answer(question, context)
        print(f"Answer : {res['text']}")
        print(f"Span   : chars [{res['char_start']}:{res['char_end']}], score={res['score']:.2f}")
        print(f"Context: {highlight_span(context, res['char_start'], res['char_end'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini Extractive Document QA System (LAB 10)")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("LAB10_QA_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Path to fine-tuned QA checkpoint directory.",
    )
    parser.add_argument("--question", type=str, default=None, help="Question to ask.")
    parser.add_argument("--context", type=str, default=None, help="Context passage text.")
    parser.add_argument("--context-file", type=str, default=None, help="Path to a text file as context.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive QA prompt.")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo questions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa = DocumentQASystem(model_dir=args.model_dir)

    if args.interactive:
        run_interactive(qa)
        return

    if args.question:
        if args.context_file:
            context = Path(args.context_file).read_text(encoding="utf-8")
        elif args.context:
            context = args.context
        else:
            raise ValueError("Please provide either --context or --context-file with --question.")
        res = qa.answer(args.question, context)
        print(f"Question : {res['question']}")
        print(f"Answer   : {res['text']}")
        print(f"CharSpan : ({res['char_start']}, {res['char_end']})")
        print(f"Score    : {res['score']:.2f}")
        print(f"Snippet  : {highlight_span(context, res['char_start'], res['char_end'])}")
        return

    # Default action: run demo
    run_demo(qa)


if __name__ == "__main__":
    main()
