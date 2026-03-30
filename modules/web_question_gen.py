"""
modules/web_question_gen.py
---------------------------
Builds factual questions from web-fetched topic summaries.
Optimized for people, achievements, dates, roles, and places.
"""

from __future__ import annotations

import random
import re

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b")
ROLE_PATTERN = re.compile(r"\b(?:is|was)\s+an?\s+([^.,;]+)")
PLACE_PATTERN = re.compile(r"\b(?:born in|from|in)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)")


def _sentences(text: str) -> list[str]:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if 35 <= len(part.strip()) <= 240
    ]
    return sentences[:10]


def _entity_options(correct: str, pool: list[str]) -> list[str]:
    options = [correct]
    for item in pool:
        if item.lower() != correct.lower() and item.lower() not in {o.lower() for o in options}:
            options.append(item)
        if len(options) == 4:
            break
    random.shuffle(options)
    return options[:4]


def _date_options(answer: str) -> list[str]:
    if YEAR_PATTERN.fullmatch(answer):
        value = int(answer)
        options = [answer, str(value - 1), str(value + 1), str(value + 5)]
    else:
        year = YEAR_PATTERN.search(answer)
        if year:
            value = int(year.group(0))
            options = [answer, answer.replace(str(value), str(value - 1)), answer.replace(str(value), str(value + 1)), answer.replace(str(value), str(value + 5))]
        else:
            options = [answer, "1 January 2000", "7 July 1981", "15 August 1990"]
    random.shuffle(options)
    return options[:4]


def generate_web_questions(topic: dict, num_questions: int, question_types: list[str]) -> list[dict]:
    title = topic["title"]
    text = topic["text"]
    summary = topic.get("summary", text)
    sentences = _sentences(text)
    entities = list(dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", text)))[:20]
    questions: list[dict] = []

    birth_sentence = next((s for s in sentences if "born" in s.lower() or "birth" in s.lower()), "")
    achievement_sentences = [s for s in sentences if any(k in s.lower() for k in ["won", "award", "record", "captain", "achievement", "known for"])]
    role_sentence = next((s for s in sentences if ROLE_PATTERN.search(s)), "")

    if birth_sentence:
        birth_date = DATE_PATTERN.search(birth_sentence) or YEAR_PATTERN.search(birth_sentence)
        birth_place = PLACE_PATTERN.search(birth_sentence)
        if birth_date:
            questions.append({
                "question": f"When was {title} born?",
                "type": "Multiple Choice" if "Multiple Choice" in question_types else "Short Answer",
                "answer": birth_date.group(0),
                "options": _date_options(birth_date.group(0)) if "Multiple Choice" in question_types else [],
                "context": birth_sentence,
            })
        if birth_place:
            place = birth_place.group(1)
            place_pool = [e for e in entities if e != title]
            questions.append({
                "question": f"Where was {title} born?",
                "type": "Multiple Choice" if "Multiple Choice" in question_types else "Short Answer",
                "answer": place,
                "options": _entity_options(place, place_pool + ["Ranchi", "Mumbai", "Delhi", "Chennai"]) if "Multiple Choice" in question_types else [],
                "context": birth_sentence,
            })

    if role_sentence:
        role_match = ROLE_PATTERN.search(role_sentence)
        if role_match:
            role = role_match.group(1).strip()
            questions.append({
                "question": f"What is {title} known for?",
                "type": "Multiple Choice" if "Multiple Choice" in question_types else "Short Answer",
                "answer": role,
                "options": _entity_options(role, ["cricketer", "actor", "politician", "scientist", "writer"]) if "Multiple Choice" in question_types else [],
                "context": role_sentence,
            })

    for sentence in achievement_sentences:
        if len(questions) >= num_questions:
            break
        questions.append({
            "question": f"What achievement of {title} is mentioned?",
            "type": "Short Answer" if "Short Answer" in question_types else "Multiple Choice",
            "answer": sentence,
            "options": _entity_options(sentence, achievement_sentences) if "Multiple Choice" in question_types and "Short Answer" not in question_types else [],
            "context": sentence,
        })

    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        year_match = YEAR_PATTERN.search(sentence)
        if year_match:
            questions.append({
                "question": f"Which year is mentioned in this fact about {title}?",
                "type": "Multiple Choice" if "Multiple Choice" in question_types else "Short Answer",
                "answer": year_match.group(0),
                "options": _date_options(year_match.group(0)) if "Multiple Choice" in question_types else [],
                "context": sentence,
            })

    if not questions:
        questions.append({
            "question": f"What key fact is stated about {title}?",
            "type": "Short Answer",
            "answer": summary,
            "context": summary,
        })

    normalized = []
    for idx, item in enumerate(questions[:num_questions]):
        q_type = item["type"]
        if q_type not in question_types:
            q_type = question_types[0]
        normalized.append(
            {
                "question": item["question"],
                "type": q_type,
                "answer": item["answer"],
                "options": item.get("options", []) if q_type == "Multiple Choice" else [],
                "context": item["context"],
            }
        )
    return normalized[:num_questions]
