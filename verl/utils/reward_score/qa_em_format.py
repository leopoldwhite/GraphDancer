# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text: str):
    tag_re = re.compile(r"</?(think|graph|information|answer)>", re.IGNORECASE)

    state = "start"
    i = 0
    n = len(text)
    in_information = False

    while i <= n:
        m = tag_re.search(text, i)
        if not m:
            remainder = text[i:]
            if in_information:
                return False, "Incomplete sequence, missing </information>"
            if remainder.strip():
                return False, f"Unexpected content '{remainder.strip()[:40]}' between tags (state: {state})"
            break

        tag = m.group(0).lower()
        before = text[i:m.start()]

        if in_information:
            if tag != "</information>":
                i = m.end()
                continue
            in_information = False
            state = "information"
            i = m.end()
            continue

        if before.strip():
            if state not in ["in_think", "in_graph", "in_answer"]:
                return False, f"Unexpected content '{before.strip()[:40]}' between tags (state: {state})"

        if tag == "<think>":
            if state not in ["start", "information"]:
                return False, f"Unexpected tag {tag} in state {state}"
            state = "in_think"
        elif tag == "</think>":
            if state != "in_think":
                return False, f"Unexpected tag {tag} in state {state}"
            state = "after_think"
        elif tag == "<graph>":
            if state != "after_think":
                return False, f"Unexpected tag {tag} in state {state}"
            state = "in_graph"
        elif tag == "</graph>":
            if state != "in_graph":
                return False, f"Unexpected tag {tag} in state {state}"
            state = "after_graph"
        elif tag == "<information>":
            if state not in ["after_graph", "information"]:
                return False, f"Unexpected tag {tag} in state {state}"
            in_information = True
            state = "in_information"
        elif tag == "<answer>":
            if state != "after_think":
                return False, f"Unexpected tag {tag} in state {state}"
            state = "in_answer"
        elif tag == "</answer>":
            if state != "in_answer":
                return False, f"Unexpected tag {tag} in state {state}"
            state = "end"

        i = m.end()

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method='strict',
    structure_format_score=0.2,
    final_format_score=0.1,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    """EM reward with structural/retrieval components.

    Returns a float combining correctness, structural format, and retrieval signal.
    """
    is_valid_format, error_message = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 10) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"Format message: {error_message}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                if do_print:
                    print(f"Reward: structure_format_score + retrieval_score {structure_format_score + retrieval_score}")
                return structure_format_score + retrieval_score
            else:
                if do_print:
                    print(f"Reward: structure_format_score {structure_format_score}")
                return structure_format_score
        else:
            if do_print:
                print(f"Reward: 0")
            return 0
    else:
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                if do_print:
                    print(f"Reward: score {score}")
                return score
            else:
                if do_print:
                    print(f"Reward: score - structure_format_score {score - structure_format_score}")
                return score - structure_format_score
        elif is_valid_format:
            if retrieval_correct:
                if do_print:
                    print(f"Reward: structure_format_score + retrieval_score {structure_format_score + retrieval_score}")
                return structure_format_score + retrieval_score
            else:
                if do_print:
                    print(f"Reward: structure_format_score {structure_format_score}")
                return structure_format_score
        else:
            if do_print:
                print(f"Reward: final_format_score {final_format_score}")
            return final_format_score

