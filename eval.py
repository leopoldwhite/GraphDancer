import argparse
import json
import os
import sys

import evaluate
from openai import OpenAI


def compute_exact_match(predictions, references):
    try:
        em_metric = evaluate.load("exact_match")
    except ImportError as e:
        print(f"Warning: exact_match metric unavailable: {e}", file=sys.stderr)
        return None
    return em_metric.compute(predictions=predictions, references=references)


def compute_bleu(predictions, references):
    try:
        bleu_metric = evaluate.load("bleu")
    except ImportError as e:
        print(f"Warning: bleu metric unavailable: {e}", file=sys.stderr)
        return None
    return bleu_metric.compute(predictions=predictions, references=references)


def compute_rouge(predictions, references):
    try:
        rouge_metric = evaluate.load("rouge")
    except ImportError as e:
        print(f"Warning: rouge metric unavailable: {e}", file=sys.stderr)
        return None
    return rouge_metric.compute(predictions=predictions, references=references)


def gpt4_score(predictions, references, questions, openai_key: str | None):
    if not openai_key or openai_key == "None":
        return None

    client = OpenAI(api_key=openai_key)
    eval_prompt = (
        "Question:{} \nModel prediction: {} \nGround truth: {}. \n"
        "Please help me judge if the model prediction is correct or not "
        "given the question and ground truth answer. "
        "Please use one word (Yes or No) to answer. Do not explain."
    )

    res = []
    for pred, ref, question in zip(predictions, references, questions, strict=True):
        x = eval_prompt.format(question, pred, ref)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a generative language model evaluator."},
                {"role": "user", "content": x},
            ],
            temperature=0.01,
            top_p=1.0,
        )
        gpt_judgement = response.choices[0].message.content
        if gpt_judgement not in ["Yes", "No"]:
            # Fallback: treat non-Yes as incorrect
            res.append(0)
        else:
            res.append(1 if gpt_judgement == "Yes" else 0)
    return sum(res) / len(res) if res else None


def read_jsonl(file_path: str):
    results = []
    preds = []
    gts = []
    questions = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            tmp = json.loads(line)
            results.append(tmp)
            preds.append(tmp.get("model_answer"))
            gts.append(tmp.get("gt_answer"))
            questions.append(tmp.get("question"))
    return results, preds, gts, questions


def resolve_result_file(result_file: str | None, model: str | None, graph_name: str | None) -> str:
    """
    Resolve the path to the result file.

    - If result_file is provided (and not 'None'), use it directly.
    - Otherwise, expect model and graph_name, and look for:
        results/<model_name>/<graph_name>_results.json
      where <model_name> is usually MODEL_ID##*/ from training.
    """
    if result_file and result_file != "None":
        return result_file

    if not model or model == "None":
        raise ValueError("Either --result_file or --model must be provided.")
    if not graph_name or graph_name == "None":
        raise ValueError("When --result_file is not provided, --graph_name must be set.")

    model_name = os.path.basename(model.rstrip("/"))
    base_dir = os.path.join("results", model_name)
    return os.path.join(base_dir, f"{graph_name}_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA-style generations (EM / BLEU / ROUGE / optional GPT4score)."
    )
    parser.add_argument("--result_file", type=str, default=None, help="Path to a results JSONL file.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (used in summary print and default results path).",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default=None,
        help="Graph/dataset name. Used when --result_file is not provided.",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, GPT4score is skipped.",
    )
    args = parser.parse_args()

    result_path = resolve_result_file(args.result_file, args.model, args.graph_name)
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    results, preds, gts, questions = read_jsonl(result_path)
    preds = [pred if pred is not None else "" for pred in preds]

    em = compute_exact_match(preds, gts)
    bleu = compute_bleu(preds, gts)
    rouge = compute_rouge(preds, gts)
    gpt4 = gpt4_score(preds, gts, questions, args.openai_key)

    model_name = args.model or os.path.basename(result_path)

    # Derive graph_name if not provided explicitly
    graph_name = args.graph_name
    if graph_name in (None, "None"):
        base = os.path.basename(result_path)
        if base.endswith("_results.json"):
            graph_name = base[: -len("_results.json")]

    metrics = {
        "EM": em["exact_match"] if em is not None else None,
        "Bleu": bleu["bleu"] if bleu is not None else None,
        "Rouge1": rouge["rouge1"] if rouge is not None else None,
        "Rouge2": rouge["rouge2"] if rouge is not None else None,
        "RougeL": rouge["rougeL"] if rouge is not None else None,
        "RougeLSum": rouge["rougeLsum"] if rouge is not None else None,
        "GPT4score": gpt4,
    }

    output = {
        "model_name": model_name,
        "graph_name": graph_name,
        "metrics": metrics,
    }

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
