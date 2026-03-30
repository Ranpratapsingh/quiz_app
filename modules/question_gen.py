"""
modules/question_gen.py
-----------------------
Generates quiz questions from document chunks using a local Ollama LLM.

Supports three question types:
  - Multiple Choice (4 options, one correct)
  - True/False
  - Short Answer

Questions are generated in batches per chunk, then deduplicated and trimmed
to the requested count. The defaults favor speed for local Ollama models.
"""

import random
import re
from modules.ollama_client import chat, parse_json_response

PROMPT_CHARS_FAST = 1200
PROMPT_CHARS_STANDARD = 1800
STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "before", "being",
    "between", "could", "during", "first", "from", "have", "into", "its",
    "many", "more", "most", "other", "over", "same", "such", "than", "that",
    "their", "there", "these", "they", "this", "those", "through", "using",
    "what", "when", "where", "which", "while", "with", "would", "your",
}

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9&\-]{2,}(?:\s+[A-Z][A-Za-z0-9&\-]{2,})*\b")
PLACE_PATTERN = re.compile(r"\b(?:in|at|from)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)")


def _sentence_candidates(text: str) -> list[str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if 40 <= len(s.strip()) <= 220
    ]
    return sentences[:12]


def _extract_focus_value(sentence: str) -> tuple[str, str] | None:
    year_match = YEAR_PATTERN.search(sentence)
    if year_match:
        return ("year", year_match.group(0))

    number_matches = NUMBER_PATTERN.findall(sentence)
    if number_matches:
        for value in number_matches:
            if len(value) >= 2:
                return ("stat", value)

    return None


def _keyword_pool(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", text)
    freq: dict[str, int] = {}
    original: dict[str, str] = {}
    for token in tokens:
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        freq[lower] = freq.get(lower, 0) + 1
        original.setdefault(lower, token)
    ranked = sorted(freq.items(), key=lambda item: (-item[1], -len(item[0])))
    return [original[word] for word, _ in ranked[:20]]


def _entity_pool(text: str) -> list[str]:
    entities = []
    seen = set()
    for match in ENTITY_PATTERN.findall(text):
        cleaned = match.strip()
        lower = cleaned.lower()
        if lower in STOPWORDS or len(cleaned) < 3:
            continue
        if lower not in seen:
            seen.add(lower)
            entities.append(cleaned)
    return entities[:20]


def _pick_answer_term(sentence: str, keyword_pool: list[str], entity_pool: list[str]) -> str:
    sentence_words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", sentence)
    sentence_lookup = {word.lower(): word for word in sentence_words}

    for entity in entity_pool:
        if entity.lower() in sentence.lower():
            return entity

    for keyword in keyword_pool:
        if keyword.lower() in sentence_lookup:
            return sentence_lookup[keyword.lower()]

    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", sentence)
    if capitalized:
        return capitalized[0]

    return sentence_words[0] if sentence_words else sentence.split()[0]


def _build_distractors(answer: str, keyword_pool: list[str], entity_pool: list[str]) -> list[str]:
    distractors = []
    pools = entity_pool + keyword_pool
    for keyword in pools:
        if keyword.lower() == answer.lower():
            continue
        if keyword.lower() in {d.lower() for d in distractors}:
            continue
        distractors.append(keyword)
        if len(distractors) == 3:
            break

    fallback_terms = ["analysis", "dataset", "process", "system", "method", "result"]
    for term in fallback_terms:
        if term.lower() != answer.lower() and term.lower() not in {d.lower() for d in distractors}:
            distractors.append(term.title())
        if len(distractors) == 3:
            break

    return distractors[:3]


def _build_mcq_from_sentence(sentence: str, keyword_pool: list[str], entity_pool: list[str]) -> dict:
    person = _primary_entity(sentence, entity_pool)
    life_question = _build_personal_question(sentence, person)
    if life_question:
        return _build_mcq_from_prompt(life_question["question"], life_question["answer"], life_question["distractors"], sentence)

    focus = _extract_focus_value(sentence)
    if focus:
        focus_type, answer = focus
        distractors = _build_numeric_distractors(answer, sentence)
        prompt = (
            f"Which {focus_type} is mentioned in this statement?\n\n{sentence}"
            if focus_type == "year"
            else f"What value or statistic is mentioned in this statement?\n\n{sentence}"
        )
    else:
        answer = _pick_answer_term(sentence, keyword_pool, entity_pool)
        distractors = _build_distractors(answer, keyword_pool, entity_pool)
        prompt = (
            f"Which term best answers this question about the text?\n\n"
            f"What is the key subject mentioned here: {sentence}"
        )

    return _build_mcq_from_prompt(prompt, answer, distractors, sentence)


def _build_short_answer_from_sentence(sentence: str) -> dict:
    person = _primary_entity(sentence, [])
    life_question = _build_personal_question(sentence, person)
    if life_question:
        return {
            "question": life_question["question"],
            "type": "Short Answer",
            "answer": life_question["answer"],
            "context": sentence,
        }

    focus = _extract_focus_value(sentence)
    if focus:
        focus_type, answer = focus
        question = (
            f"What year is mentioned here?\n\n{sentence}"
            if focus_type == "year"
            else f"What statistic or number is mentioned here?\n\n{sentence}"
        )
        return {
            "question": question,
            "type": "Short Answer",
            "answer": answer,
            "context": sentence,
        }

    snippet = sentence[:1].lower() + sentence[1:]
    return {
        "question": f"What key fact is stated about this idea?\n\n{snippet}",
        "type": "Short Answer",
        "answer": sentence,
        "context": sentence,
    }


def _build_mcq_from_prompt(question: str, answer: str, distractors: list[str], sentence: str) -> dict:
    options = [answer] + distractors
    deduped = []
    seen = set()
    for option in options:
        normalized = option.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(option)
    random.shuffle(deduped)
    return {
        "question": question,
        "type": "Multiple Choice",
        "options": deduped[:4],
        "answer": answer,
        "context": sentence,
    }


def _primary_entity(sentence: str, entity_pool: list[str]) -> str:
    for entity in entity_pool:
        if entity.lower() in sentence.lower():
            return entity
    entities = ENTITY_PATTERN.findall(sentence)
    return entities[0] if entities else ""


def _build_personal_question(sentence: str, person: str) -> dict | None:
    lower_sentence = sentence.lower()
    if not person:
        return None

    year_match = YEAR_PATTERN.search(sentence)
    place_match = PLACE_PATTERN.search(sentence)

    if "born" in lower_sentence or "birth" in lower_sentence:
        if year_match:
            return {
                "question": f"When was {person} born?",
                "answer": year_match.group(0),
                "distractors": _build_numeric_distractors(year_match.group(0), sentence),
            }
        if place_match:
            place = place_match.group(1)
            return {
                "question": f"Where was {person} born?",
                "answer": place,
                "distractors": _build_place_distractors(place, sentence),
            }

    if any(keyword in lower_sentence for keyword in ["won", "award", "achieve", "achievement", "captain", "record"]):
        answer = sentence.strip()
        return {
            "question": f"What achievement is mentioned for {person}?",
            "answer": answer,
            "distractors": _generic_fact_distractors(answer),
        }

    if year_match and person:
        return {
            "question": f"In which year is {person} mentioned in this fact?",
            "answer": year_match.group(0),
            "distractors": _build_numeric_distractors(year_match.group(0), sentence),
        }

    if person:
        return {
            "question": f"What is mentioned about {person}?",
            "answer": sentence.strip(),
            "distractors": _generic_fact_distractors(sentence.strip()),
        }

    return None


def _build_place_distractors(answer: str, sentence: str) -> list[str]:
    places = [match.group(1) for match in PLACE_PATTERN.finditer(sentence)]
    distractors = []
    for place in places:
        if place.lower() != answer.lower() and place.lower() not in {d.lower() for d in distractors}:
            distractors.append(place)
    for fallback in ["Ranchi", "Mumbai", "Delhi", "Chennai"]:
        if fallback.lower() != answer.lower() and fallback.lower() not in {d.lower() for d in distractors}:
            distractors.append(fallback)
        if len(distractors) == 3:
            break
    return distractors[:3]


def _generic_fact_distractors(answer: str) -> list[str]:
    fallback = [
        "A different personal milestone is described.",
        "A separate award or role is mentioned.",
        "The text refers to another event entirely.",
    ]
    return [item for item in fallback if item != answer][:3]


def _build_numeric_distractors(answer: str, sentence: str) -> list[str]:
    distractors: list[str] = []
    if answer.endswith("%"):
        base = answer.rstrip("%")
        if base.isdigit():
            value = int(base)
            candidates = [f"{max(value - 10, 1)}%", f"{value + 5}%", f"{value + 10}%"]
            distractors.extend(candidates)
    elif answer.isdigit():
        value = int(answer)
        step = 1 if value < 10 else (5 if value < 100 else 10)
        distractors.extend([str(max(value - step, 1)), str(value + step), str(value + step * 2)])
    else:
        year_match = YEAR_PATTERN.fullmatch(answer)
        if year_match:
            value = int(answer)
            distractors.extend([str(value - 1), str(value + 1), str(value + 5)])

    if len(distractors) < 3:
        sentence_numbers = [n for n in NUMBER_PATTERN.findall(sentence) if n != answer]
        for value in sentence_numbers:
            if value not in distractors:
                distractors.append(value)
            if len(distractors) == 3:
                break

    fallback = ["5", "10", "15", "20", "2020", "2021", "2022"]
    for item in fallback:
        if item != answer and item not in distractors:
            distractors.append(item)
        if len(distractors) == 3:
            break

    return distractors[:3]


def generate_questions_locally(
    chunks: list[str],
    num_questions: int,
    question_types: list[str],
) -> list[dict]:
    """
    Fast non-LLM fallback so the app remains usable if Ollama is slow.
    Builds simple extractive questions from the best chunk.
    """
    if not chunks:
        return []

    base_text = chunks[0][:1600]
    candidates = _sentence_candidates(base_text)
    if not candidates:
        candidates = [base_text[:240].strip()]
    keyword_pool = _keyword_pool(base_text)
    entity_pool = _entity_pool(base_text)

    questions: list[dict] = []
    for idx, sentence in enumerate(candidates):
        q_type = question_types[idx % len(question_types)]
        if q_type == "Multiple Choice":
            questions.append(_build_mcq_from_sentence(sentence, keyword_pool, entity_pool))
        else:
            questions.append(_build_short_answer_from_sentence(sentence))
        if len(questions) >= num_questions:
            break

    return questions


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert quiz creator. Your task is to generate clear,
accurate quiz questions directly based on the provided text.

Rules:
- Questions must be answerable ONLY from the provided text
- Do NOT ask questions about information not present in the text
- Make questions specific, not vague
- Avoid trivial or overly simple questions
- Prefer questions about dates, years, names, counts, percentages, and concrete facts when available
- Avoid repetitive openings like "According to the passage" or "According to the text"
- For multiple choice: provide exactly 4 options, only one correct
- For short answer: expect a 1-3 sentence response
- Always respond with valid JSON only — no explanations, no markdown prose outside JSON
"""


# ── Prompt templates ───────────────────────────────────────────────────────────

def build_prompt(
    chunk: str,
    q_types: list[str],
    num_q: int,
    difficulty: str,
    max_context_chars: int,
) -> str:
    type_instructions = []

    if "Multiple Choice" in q_types:
        type_instructions.append(
            '  - "Multiple Choice": include "options" (list of 4 strings) and "answer" (the correct option text)'
        )
    if "True/False" in q_types:
        type_instructions.append(
            '  - "True/False": "answer" must be exactly "True" or "False"'
        )
    if "Short Answer" in q_types:
        type_instructions.append(
            '  - "Short Answer": "answer" is a 1-3 sentence model answer'
        )

    type_str = "\n".join(type_instructions)
    type_list = ", ".join(f'"{t}"' for t in q_types)

    return f"""Read the following text carefully and generate exactly {num_q} quiz questions at {difficulty} difficulty.

TEXT:
\"\"\"
{chunk[:max_context_chars]}
\"\"\"

Generate {num_q} questions. Mix types from: {type_list}.

Return ONLY a JSON array. Each element must have these fields:
- "question": the question string
- "type": one of {type_list}
- "answer": the correct answer
- "options": (only for Multiple Choice) list of 4 answer strings
- "context": a 1-sentence excerpt from the text that supports the answer

Type-specific rules:
{type_str}

Difficulty guide for {difficulty}:
- Easy: factual recall, direct quotes from text
- Medium: comprehension, cause-and-effect, definitions
- Hard: inference, analysis, application, multi-step reasoning

Question writing preferences:
- Prefer dates, years, names, counts, percentages, and concrete facts when the text includes them
- Avoid repetitive openings like "According to the passage" or "According to the text"

Output format example:
[
  {{
    "question": "What is the main topic discussed?",
    "type": "Short Answer",
    "answer": "The main topic is ...",
    "context": "The text states that ..."
  }},
  {{
    "question": "Which of the following is correct?",
    "type": "Multiple Choice",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option B",
    "context": "According to the text ..."
  }}
]

Respond with ONLY the JSON array. No preamble, no explanation."""


# ══════════════════════════════════════════════════════════════════════════════
#  Core generation logic
# ══════════════════════════════════════════════════════════════════════════════

def generate_from_chunk(
    chunk: str,
    model: str,
    num_q: int,
    q_types: list[str],
    difficulty: str,
    fast_mode: bool = True,
) -> list[dict]:
    """Generate questions from a single chunk. Returns list of question dicts."""
    prompt = build_prompt(
        chunk,
        q_types,
        num_q,
        difficulty,
        max_context_chars=PROMPT_CHARS_FAST if fast_mode else PROMPT_CHARS_STANDARD,
    )

    try:
        raw = chat(
            model=model,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2 if fast_mode else 0.35,
            max_tokens=900 if fast_mode else 1400,
            timeout=12 if fast_mode else 120,
        )
        questions = parse_json_response(raw)

        if not isinstance(questions, list):
            return []

        # Validate and normalise each question
        valid = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            if not q.get("question") or not q.get("answer"):
                continue

            q_type = q.get("type", "Short Answer")

            # Ensure type is one we asked for
            if q_type not in q_types:
                q["type"] = random.choice(q_types)

            # Normalise MCQ options
            if q_type == "Multiple Choice":
                opts = q.get("options", [])
                if len(opts) < 2:
                    # Fall back to short answer if options are missing
                    q["type"] = "Short Answer"
                else:
                    # Make sure correct answer is in options
                    if q["answer"] not in opts:
                        opts[0] = q["answer"]
                    q["options"] = opts[:4]

            # Ensure True/False answer is canonical
            if q_type == "True/False":
                ans_lower = str(q["answer"]).strip().lower()
                q["answer"] = "True" if ans_lower in ("true", "yes", "1") else "False"

            q["context"] = q.get("context", "")
            valid.append(q)

        return valid

    except Exception as e:
        # Log but don't crash — we'll try other chunks
        print(f"[question_gen] Chunk generation error: {e}")
        return []


def deduplicate_questions(questions: list[dict]) -> list[dict]:
    """Remove near-duplicate questions (same first 60 chars)."""
    seen = set()
    unique = []
    for q in questions:
        key = q["question"][:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_questions(
    chunks: list[str],
    model: str,
    num_questions: int = 5,
    question_types: list[str] | None = None,
    difficulty: str = "Medium",
    fast_mode: bool = True,
) -> list[dict]:
    """
    Generate `num_questions` quiz questions from a list of text chunks.

    Strategy:
      1. Distribute questions across chunks evenly
      2. Generate slightly more than needed (buffer for failures/deduplication)
      3. Deduplicate and trim to exact count

    Args:
        chunks:         List of text chunks from ingest_document()
        model:          Ollama model name
        num_questions:  Total questions to return
        question_types: List of enabled types; defaults to all three
        difficulty:     "Easy" | "Medium" | "Hard"

    Returns:
        List of question dicts with keys: question, type, answer, options?, context
    """
    if question_types is None:
        question_types = ["Multiple Choice", "True/False", "Short Answer"]

    if not chunks:
        raise ValueError("No chunks provided. Run ingestion first.")

    if not question_types:
        raise ValueError("At least one question type must be selected.")

    # Fast mode skips Ollama entirely so the UI stays responsive.
    if fast_mode:
        fallback = generate_questions_locally(chunks, num_questions, question_types)
        if fallback:
            return fallback[:num_questions]
        raise RuntimeError("Could not create local questions from the provided content.")

    # Standard mode uses the most content-rich chunks with Ollama.
    sorted_chunks = sorted(chunks, key=len, reverse=True)
    max_chunks = 4
    use_chunks = sorted_chunks[:min(max_chunks, len(sorted_chunks))]

    target = num_questions + max(2, num_questions // 2)
    q_per_chunk = max(1, (target + len(use_chunks) - 1) // len(use_chunks))

    all_questions: list[dict] = []

    for i, chunk in enumerate(use_chunks):
        if len(all_questions) >= target:
            break

        remaining = target - len(all_questions)
        batch_size = min(q_per_chunk, remaining)

        print(f"[question_gen] Chunk {i+1}/{len(use_chunks)}: generating {batch_size} questions...")

        batch = generate_from_chunk(
            chunk=chunk,
            model=model,
            num_q=batch_size,
            q_types=question_types,
            difficulty=difficulty,
            fast_mode=False,
        )
        all_questions.extend(batch)

    # Deduplicate and shuffle
    unique = deduplicate_questions(all_questions)
    random.shuffle(unique)

    if not unique:
        fallback = generate_questions_locally(use_chunks or chunks, num_questions, question_types)
        if fallback:
            return fallback
        raise RuntimeError(
            "The LLM did not generate any valid questions. "
            "Try a different model or document."
        )

    trimmed = unique[:num_questions]
    if len(trimmed) >= num_questions:
        return trimmed

    fallback = generate_questions_locally(use_chunks or chunks, num_questions, question_types)
    if fallback:
        return fallback[:num_questions]

    raise RuntimeError(
        f"Only {len(trimmed)} valid questions were generated. "
        "Try a shorter document, fewer questions, or disable Fast Mode."
    )
