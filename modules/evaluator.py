"""
modules/evaluator.py
--------------------
Evaluates user answers using the local Ollama LLM.

Uses semantic evaluation — not simple string matching.
Returns a score (0-10), a correctness flag, and actionable feedback.
"""

from modules.ollama_client import chat, parse_json_response


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a fair and strict quiz evaluator.
Your task is to evaluate a student's answer against the correct answer.

Scoring guide:
  10 = Completely correct, all key points covered
  8-9 = Mostly correct, minor omissions or imprecision
  6-7 = Partially correct, main idea present but missing important details
  4-5 = Marginally correct, shows some understanding but significant gaps
  2-3 = Mostly wrong but shows a tiny bit of relevant knowledge
  0-1 = Completely wrong or no answer given

Rules:
- Be strict but fair
- Accept paraphrases and synonyms as correct
- For Multiple Choice/True-False: score is 10 (correct) or 0 (wrong), nothing in between
- For Short Answer: use the full 0-10 scale
- Feedback must be 1-2 sentences, constructive, and specific
- Always respond with valid JSON only
"""


def build_eval_prompt(
    question: str,
    user_answer: str,
    correct_answer: str,
    context: str,
    q_type: str = "Short Answer",
) -> str:
    ctx_block = f'\nContext from document:\n"{context}"' if context else ""

    return f"""Evaluate this quiz answer:

Question: {question}
Question type: {q_type}
Correct answer: {correct_answer}{ctx_block}

Student's answer: "{user_answer or "(no answer given)"}"

Return a JSON object with exactly these fields:
{{
  "score": <integer 0-10>,
  "is_correct": <true if score >= 7, false otherwise>,
  "feedback": "<1-2 sentence feedback for the student>"
}}

Only return the JSON object. No other text."""


# ══════════════════════════════════════════════════════════════════════════════
#  Special-case evaluators (fast path, no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_true_false_fast(user_answer: str, correct_answer: str) -> dict:
    """Evaluate True/False without calling the LLM."""
    user_norm    = str(user_answer).strip().lower()
    correct_norm = str(correct_answer).strip().lower()

    is_correct = (
        (user_norm in ("true", "t", "yes", "1") and correct_norm == "true") or
        (user_norm in ("false", "f", "no", "0") and correct_norm == "false")
    )

    return {
        "score": 10 if is_correct else 0,
        "is_correct": is_correct,
        "feedback": (
            "Correct! Well done." if is_correct
            else f"Incorrect. The correct answer is {correct_answer}."
        ),
    }


def evaluate_mcq_fast(user_answer: str, correct_answer: str) -> dict:
    """Evaluate MCQ without calling the LLM (exact match on selected option)."""
    is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
    return {
        "score": 10 if is_correct else 0,
        "is_correct": is_correct,
        "feedback": (
            "Correct choice!" if is_correct
            else f"Incorrect. The correct answer was: {correct_answer}"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_answer(
    question: str,
    user_answer: str,
    correct_answer: str,
    context: str = "",
    model: str = "llama3",
    q_type: str = "Short Answer",
) -> dict:
    """
    Evaluate a user's answer and return a scoring dict.

    For MCQ and True/False: uses fast local logic (no LLM call needed).
    For Short Answer: calls Ollama for semantic evaluation.

    Args:
        question:       The quiz question text
        user_answer:    What the student typed/selected
        correct_answer: The expected correct answer
        context:        Source sentence from the document (helps LLM evaluate)
        model:          Ollama model name
        q_type:         "Multiple Choice" | "True/False" | "Short Answer"

    Returns:
        dict with keys: score (0-10), is_correct (bool), feedback (str)
    """
    # Handle empty answers quickly
    if not user_answer or not user_answer.strip():
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "No answer provided. Try to write something even if unsure.",
        }

    # Fast-path for objective question types
    if q_type == "True/False":
        return evaluate_true_false_fast(user_answer, correct_answer)

    if q_type == "Multiple Choice":
        return evaluate_mcq_fast(user_answer, correct_answer)

    # Short Answer — semantic LLM evaluation
    prompt = build_eval_prompt(
        question=question,
        user_answer=user_answer,
        correct_answer=correct_answer,
        context=context,
        q_type=q_type,
    )

    try:
        raw = chat(
            model=model,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1,   # Low temperature → consistent scoring
            max_tokens=300,
        )

        result = parse_json_response(raw)

        if not isinstance(result, dict):
            raise ValueError("LLM did not return a JSON object")

        score = int(result.get("score", 5))
        score = max(0, min(10, score))   # clamp to [0, 10]

        return {
            "score": score,
            "is_correct": result.get("is_correct", score >= 7),
            "feedback": result.get("feedback", "Evaluated."),
        }

    except Exception as e:
        print(f"[evaluator] Evaluation error: {e}")
        # Graceful fallback: simple keyword overlap
        return _fallback_evaluation(user_answer, correct_answer)


def _fallback_evaluation(user_answer: str, correct_answer: str) -> dict:
    """
    Simple keyword-based fallback when LLM evaluation fails.
    Computes Jaccard similarity on word tokens.
    """
    def tokenize(text):
        import re
        return set(re.findall(r"\b\w+\b", text.lower()))

    user_tokens    = tokenize(user_answer)
    correct_tokens = tokenize(correct_answer)

    if not correct_tokens:
        return {"score": 5, "is_correct": False, "feedback": "Could not evaluate."}

    intersection = user_tokens & correct_tokens
    union        = user_tokens | correct_tokens
    jaccard      = len(intersection) / len(union) if union else 0

    score = round(jaccard * 10)
    return {
        "score": score,
        "is_correct": score >= 7,
        "feedback": (
            "Good answer!" if score >= 7
            else f"Partial credit. Expected keywords: {', '.join(list(correct_tokens)[:5])}"
        ),
    }
