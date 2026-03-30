"""
Quiz application entrypoint.
Run with: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from modules.evaluator import evaluate_answer
from modules.ingestion import ingest_document, ingest_text
from modules.ollama_client import check_ollama, list_models
from modules.question_gen import generate_questions
from modules.session import compute_result_summary, create_session, get_all_sessions, save_session
from modules.web_question_gen import generate_web_questions
from modules.web_research import fetch_topic_context


st.set_page_config(
    page_title="Quiz App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 18%, rgba(0, 168, 150, 0.16), transparent 24%),
                radial-gradient(circle at 88% 14%, rgba(255, 166, 43, 0.24), transparent 22%),
                radial-gradient(circle at 72% 78%, rgba(224, 82, 99, 0.10), transparent 20%),
                linear-gradient(180deg, #fffaf1 0%, #eef7f5 100%);
        }
        .hero-card, .form-card, .quiz-card, .summary-card, .status-card, .upload-panel, .results-hero, .name-panel {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(16, 71, 82, 0.10);
            border-radius: 24px;
            box-shadow: 0 18px 48px rgba(32, 50, 64, 0.10);
        }
        .hero-card {
            padding: 2rem 2.2rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(246,255,252,0.92)),
                linear-gradient(120deg, #0f766e, #f4a340);
        }
        .form-card, .status-card {
            padding: 1.3rem 1.4rem;
        }
        .quiz-card {
            padding: 1.3rem 1.4rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.97), rgba(244,251,248,0.93));
        }
        .summary-card {
            padding: 1.3rem;
        }
        .metric-chip {
            display: inline-block;
            margin-right: 0.75rem;
            margin-bottom: 0.75rem;
            padding: 0.65rem 1rem;
            border-radius: 999px;
            background: #12443a;
            color: #fff7e6;
            font-weight: 600;
        }
        .good-text {
            color: #117a48;
            font-weight: 600;
        }
        .bad-text {
            color: #b23a2a;
            font-weight: 600;
        }
        .muted-text {
            color: #53635d;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.10);
            color: #0f766e;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0.9rem 0 0.55rem 0;
            font-size: 3.2rem;
            line-height: 1.02;
            color: #1f2937;
        }
        .hero-title .grad-a {
            background: linear-gradient(135deg, #0f766e, #0ea5a4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero-title .grad-b {
            background: linear-gradient(135deg, #e05263, #f59e0b);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero-copy {
            margin: 0;
            font-size: 1.08rem;
            line-height: 1.7;
            color: #52606d;
            max-width: 56rem;
        }
        .top-strip {
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin: 1.1rem 0 1.6rem 0;
        }
        .top-pill {
            padding: 0.75rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(16, 71, 82, 0.10);
            color: #23404a;
            font-weight: 600;
        }
        .section-title {
            margin: 0 0 0.3rem 0;
            color: #20323b;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .upload-panel {
            padding: 1.2rem;
            background:
                radial-gradient(circle at top right, rgba(15,118,110,0.10), transparent 36%),
                linear-gradient(180deg, rgba(255,249,240,0.95), rgba(245,255,252,0.94));
            border: 1px solid rgba(244, 163, 64, 0.28);
        }
        .upload-title {
            margin: 0 0 0.3rem 0;
            color: #9a3412;
            font-size: 1.05rem;
            font-weight: 800;
        }
        .name-panel {
            padding: 1.2rem;
            background:
                radial-gradient(circle at top left, rgba(224,82,99,0.14), transparent 34%),
                linear-gradient(180deg, rgba(94, 23, 73, 0.96), rgba(127, 29, 29, 0.90));
            border: 1px solid rgba(244, 114, 182, 0.28);
        }
        .name-panel .section-title,
        .upload-panel .section-title {
            color: #fff7ed !important;
            font-size: 1.18rem !important;
            font-weight: 900 !important;
            margin-bottom: 0.7rem !important;
        }
        .name-panel .stTextInput label p,
        .upload-panel .stFileUploader label p,
        .upload-panel label p {
            color: #fff5f7 !important;
            font-size: 1.14rem !important;
            font-weight: 900 !important;
        }
        .name-panel .muted-text,
        .upload-panel .muted-text,
        .name-panel small,
        .upload-panel small,
        .upload-panel .stCaption {
            color: rgba(255,244,246,0.90) !important;
        }
        .quiz-shell {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(16,71,82,0.08);
            border-radius: 28px;
            padding: 1.2rem 1.3rem 1.4rem 1.3rem;
            box-shadow: 0 18px 44px rgba(32, 50, 64, 0.08);
        }
        .quiz-header {
            padding: 1.4rem 1.5rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at right top, rgba(245,158,11,0.20), transparent 26%),
                linear-gradient(135deg, rgba(240,253,250,0.96), rgba(255,247,237,0.94));
            border: 1px solid rgba(15,118,110,0.10);
            margin-bottom: 1rem;
        }
        .results-hero {
            padding: 1.5rem 1.6rem;
            background:
                radial-gradient(circle at left top, rgba(14,165,164,0.16), transparent 28%),
                radial-gradient(circle at right top, rgba(245,158,11,0.18), transparent 24%),
                linear-gradient(180deg, rgba(255,255,255,0.97), rgba(247,252,250,0.95));
        }
        .results-score {
            font-size: 3rem;
            font-weight: 900;
            line-height: 1;
            color: #0f766e;
            margin: 0.3rem 0 0.4rem 0;
        }
        .answer-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(16,71,82,0.08);
            background: rgba(255,255,255,0.84);
        }
        .answer-status-box {
            margin: 0.8rem 0 0.9rem 0;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .answer-status-box.correct {
            background: linear-gradient(180deg, rgba(220,252,231,0.95), rgba(187,247,208,0.88));
            border-color: rgba(34, 197, 94, 0.30);
            color: #166534;
        }
        .answer-status-box.wrong {
            background: linear-gradient(180deg, rgba(254,226,226,0.96), rgba(254,202,202,0.90));
            border-color: rgba(239, 68, 68, 0.28);
            color: #991b1b;
        }
        .name-panel div[data-testid="stTextInput"] input,
        .upload-panel section[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.96) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 16px !important;
        }
        .name-panel div[data-testid="stTextInput"] input {
            min-height: 3.3rem !important;
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            color: #4a1636 !important;
        }
        .name-panel div[data-testid="stTextInput"] input::placeholder {
            color: #9d5c7d !important;
            font-weight: 700 !important;
        }
        .upload-panel section[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed rgba(255,255,255,0.72) !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        .upload-panel [data-testid="stFileUploaderDropzoneInstructions"] div,
        .upload-panel [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #7c2d12 !important;
            font-weight: 700 !important;
        }
        .upload-panel {
            color: #fff7ed;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


UPLOAD_DIR = Path("data/uploads")

QUESTION_TYPE_OPTIONS = ["Multiple Choice", "Short Answer"]


def ensure_directory(path: Path, label: str) -> None:
    """
    Ensure a path exists as a directory.
    Raise a clear error if a file already exists at the same location.
    """
    if path.exists() and not path.is_dir():
        raise RuntimeError(
            f"{label} path '{path}' exists as a file. Delete that file and create a folder with the same name."
        )
    path.mkdir(parents=True, exist_ok=True)


def init_state() -> None:
    defaults = {
        "stage": "setup",
        "user_name": "",
        "source_name": "",
        "source_type": "",
        "chunks": [],
        "questions": [],
        "answers": [],
        "current_q": 0,
        "session_id": None,
        "selected_model": "llama3",
        "fast_mode": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz() -> None:
    for key in [
        "stage",
        "user_name",
        "source_name",
        "source_type",
        "chunks",
        "questions",
        "answers",
        "current_q",
        "session_id",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


def store_uploaded_file(uploaded_file) -> Path:
    save_path = UPLOAD_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.getvalue())
    return save_path


def preview_source(source_mode: str, uploaded_file, pasted_text: str) -> tuple[list[str], str]:
    if source_mode == "Upload File":
        if not uploaded_file:
            raise RuntimeError("Please upload a PDF, TXT, DOCX, or CSV file.")
        path = store_uploaded_file(uploaded_file)
        return ingest_document(str(path)), uploaded_file.name

    if source_mode == "Web Topic":
        topic = fetch_topic_context(pasted_text)
        return [topic["text"]], f"{topic['title']} (web)"

    chunks = ingest_text(pasted_text)
    return chunks, "typed_paragraph.txt"


def generate_quiz(source_mode: str, uploaded_file, pasted_text: str, model: str,
                  question_types: list[str], difficulty: str, num_questions: int,
                  fast_mode: bool) -> None:
    if source_mode == "Web Topic":
        topic = fetch_topic_context(pasted_text)
        chunks = [topic["text"]]
        source_name = f"{topic['title']} (web)"
        questions = generate_web_questions(
            topic=topic,
            num_questions=num_questions,
            question_types=question_types,
        )
    else:
        chunks, source_name = preview_source(source_mode, uploaded_file, pasted_text)
        questions = generate_questions(
            chunks=chunks,
            model=model,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            fast_mode=fast_mode,
        )
    session_id = create_session(
        filename=source_name,
        num_questions=len(questions),
        model=model,
    )
    st.session_state.source_name = source_name
    st.session_state.source_type = (
        "uploaded_file" if source_mode == "Upload File"
        else "web_topic" if source_mode == "Web Topic"
        else "typed_paragraph"
    )
    st.session_state.chunks = chunks
    st.session_state.questions = questions
    st.session_state.answers = []
    st.session_state.current_q = 0
    st.session_state.session_id = session_id
    st.session_state.stage = "quiz"


def render_setup(ollama_ok: bool, model: str) -> None:
    available_models = list_models()
    recent_sessions = list(reversed(get_all_sessions()))[:3]

    st.markdown(
        """
        <div class="hero-card">
            <span class="hero-kicker">AI Quiz Studio</span>
            <h1 class="hero-title"><span class="grad-a">Turn your notes</span> into a <span class="grad-b">polished quiz experience.</span></h1>
            <p class="hero-copy">
                Upload your study material or paste a paragraph, generate Multiple Choice and
                Short Answer questions, and get a neat score summary with feedback at the end.
            </p>
            <div class="top-strip">
                <div class="top-pill">Beautiful Streamlit interface</div>
                <div class="top-pill">PDF, TXT, DOCX, CSV, Paragraph</div>
                <div class="top-pill">Score, feedback, and percentage summary</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_left, status_right = st.columns([1.2, 1])
    with status_left:
        status_text = "Ollama connected and ready." if ollama_ok else "Ollama offline. Fast Mode can still generate local questions."
        status_class = "good-text" if ollama_ok else "bad-text"
        st.markdown(
            f"""
            <div class="status-card">
                <div class="section-title">Generation Status</div>
                <div class="{status_class}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_right:
        if recent_sessions:
            latest = recent_sessions[0]
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="section-title">Latest Attempt</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#0f766e;">{latest['score_pct']}%</div>
                    <div class="muted-text">{latest.get('user_name') or 'User'} on {latest['filename']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-card">
                    <div class="section-title">Latest Attempt</div>
                    <div class="muted-text">Your latest quiz result will appear here after the first submission.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Create Your Quiz</div>
        <p class="muted-text" style="margin-top:0;">Everything is placed directly on the home page for a cleaner experience.</p>
        """,
        unsafe_allow_html=True,
    )

    top_row_left, top_row_right = st.columns([1.05, 1])
    with top_row_left:
        st.markdown('<div class="name-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter your name</div>', unsafe_allow_html=True)
        user_name = st.text_input(
            "Enter your name",
            value=st.session_state.user_name,
            placeholder="Example: Rahul Sharma",
            label_visibility="collapsed",
        )
        st.session_state.user_name = user_name
        st.markdown("</div>", unsafe_allow_html=True)
    with top_row_right:
        if available_models:
            default_index = 0
            if st.session_state.selected_model in available_models:
                default_index = available_models.index(st.session_state.selected_model)
            st.session_state.selected_model = st.selectbox(
                "Choose Ollama model",
                options=available_models,
                index=default_index,
                help="Used when Fast Mode is turned off.",
            )
        else:
            st.session_state.selected_model = st.text_input(
                "Choose Ollama model",
                value=st.session_state.selected_model,
                help="Used when Fast Mode is turned off.",
            )

    source_mode = st.radio(
        "Select one input method",
        options=["Upload File", "Write Paragraph", "Web Topic"],
        horizontal=True,
    )

    uploaded_file = None
    pasted_text = ""

    content_col, settings_col = st.columns([1.15, 0.95])
    with content_col:
        if source_mode == "Upload File":
            st.markdown(
                """
                <div class="upload-panel">
                    <div class="section-title">Upload your PDF or document</div>
                    <div class="muted-text" style="margin-bottom:0.8rem;">
                        Upload your PDF first for the best document-based quiz flow. TXT, DOCX, and CSV also work.
                    </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload PDF, TXT, DOCX, or CSV",
                type=["pdf", "txt", "docx", "csv"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                st.caption(f"Selected file: {uploaded_file.name}")
            st.markdown("</div>", unsafe_allow_html=True)
        elif source_mode == "Write Paragraph":
            pasted_text = st.text_area(
                "Write or paste a paragraph",
                placeholder="Paste your study material here...",
                height=240,
            )
        else:
            st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
            pasted_text = st.text_input(
                "Enter a topic to fetch from web sources",
                placeholder="Example: MS Dhoni",
            )
            st.caption("Use a person, place, event, or topic name to fetch factual web context.")
            st.markdown("</div>", unsafe_allow_html=True)

    with settings_col:
        num_questions = st.slider("Number of questions", min_value=3, max_value=15, value=5)
        difficulty = st.selectbox("Difficulty", options=["Easy", "Medium", "Hard"], index=1)
        selected_question_type = st.radio(
            "Question type",
            options=QUESTION_TYPE_OPTIONS,
            horizontal=True,
        )
        question_types = [selected_question_type]
        fast_mode = st.checkbox(
            "Fast Mode",
            value=st.session_state.fast_mode,
            help="Skips Ollama for document mode. Web topic mode uses factual web questions directly.",
        )
        st.session_state.fast_mode = fast_mode

    action_left, action_right = st.columns(2)
    with action_left:
        preview_clicked = st.button("Preview Extracted Content", use_container_width=True)
    with action_right:
        generate_disabled = source_mode != "Web Topic" and (not fast_mode) and (not ollama_ok)
        generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True, disabled=generate_disabled)

    if preview_clicked:
        try:
            chunks, source_name = preview_source(source_mode, uploaded_file, pasted_text)
            st.success(f"Preview ready for {source_name}")
            st.text_area(
                "Extracted preview",
                value="\n\n---\n\n".join(chunks[:2])[:2200],
                height=220,
                disabled=True,
            )
            st.caption(f"Usable chunks created: {len(chunks)}")
        except Exception as exc:
            st.error(f"Could not preview the source: {exc}")

    if generate_clicked:
        if not user_name.strip():
            st.error("Please enter your name before generating the quiz.")
        elif not question_types:
            st.error("Please choose at least one question type.")
        else:
            spinner_label = (
                "Fetching web facts and building quiz..."
                if source_mode == "Web Topic"
                else "Generating quiz instantly..."
                if fast_mode
                else f"Generating quiz with {model}..."
            )
            with st.spinner(spinner_label):
                try:
                    generate_quiz(
                        source_mode=source_mode,
                        uploaded_file=uploaded_file,
                        pasted_text=pasted_text,
                        model=model,
                        question_types=question_types,
                        difficulty=difficulty,
                        num_questions=num_questions,
                        fast_mode=fast_mode,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Quiz generation failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_quiz(model: str) -> None:
    questions = st.session_state.questions
    current_q = st.session_state.current_q
    question = questions[current_q]
    total = len(questions)

    st.markdown('<div class="quiz-shell">', unsafe_allow_html=True)
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown(
            f"""
            <div class="quiz-header">
                <div class="section-title">Generated Quiz</div>
                <div style="font-size:2rem; font-weight:800; color:#20323b; margin-top:0.15rem;">Stay focused and answer one step at a time.</div>
                <div class="muted-text" style="margin-top:0.45rem;">
                    Student: {st.session_state.user_name} | Source: {st.session_state.source_name} | Model: {model}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Start Over", use_container_width=True):
            reset_quiz()
            st.rerun()

    st.progress((current_q + 1) / total)
    st.caption(f"Question {current_q + 1} of {total}")

    st.markdown(
        f"""
        <div class="quiz-card">
            <div style="display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; background:rgba(15,118,110,0.10); color:#0f766e; font-weight:700; margin-bottom:0.7rem;">
                Question {current_q + 1}
            </div>
            <div style="font-size:1.18rem; font-weight:800; margin-bottom:0.5rem; color:#22313f;">
                {question['question']}
            </div>
            <div class="muted-text">Type: {question.get('type', 'Short Answer')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    existing_answer = ""
    if current_q < len(st.session_state.answers):
        existing_answer = st.session_state.answers[current_q].get("user_answer", "")

    q_type = question.get("type", "Short Answer")
    if q_type == "Multiple Choice":
        options = question.get("options", [])
        if not options:
            st.warning("This MCQ has no valid options, so it has been shown as a short answer instead.")
            q_type = "Short Answer"
            user_answer = st.text_area(
                "Write your answer",
                value=existing_answer,
                height=150,
                key=f"short_{current_q}",
            )
        else:
            default_index = 0
            if existing_answer in options:
                default_index = options.index(existing_answer)
            user_answer = st.radio(
                "Choose one answer",
                options=options,
                index=default_index,
                key=f"mcq_{current_q}",
            )
    else:
        user_answer = st.text_area(
            "Write your answer",
            value=existing_answer,
            height=150,
            key=f"short_{current_q}",
        )

    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_q > 0 and st.button("Previous", use_container_width=True):
            st.session_state.current_q -= 1
            st.rerun()

    with col_next:
        button_label = "Finish Quiz" if current_q == total - 1 else "Next Question"
        if st.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Evaluating your answer..."):
                evaluation = evaluate_answer(
                    question=question["question"],
                    user_answer=user_answer or "",
                    correct_answer=question.get("answer", ""),
                    context=question.get("context", ""),
                    model=model,
                    q_type=q_type,
                )

            answer_payload = {
                "question_index": current_q,
                "question": question["question"],
                "type": q_type,
                "user_answer": user_answer or "",
                "correct_answer": question.get("answer", ""),
                "options": question.get("options", []),
                "context": question.get("context", ""),
                "score": evaluation["score"],
                "is_correct": evaluation["is_correct"],
                "feedback": evaluation["feedback"],
            }

            if current_q < len(st.session_state.answers):
                st.session_state.answers[current_q] = answer_payload
            else:
                st.session_state.answers.append(answer_payload)

            if current_q == total - 1:
                save_session(
                    session_id=st.session_state.session_id,
                    filename=st.session_state.source_name,
                    answers=st.session_state.answers,
                    model=model,
                    user_name=st.session_state.user_name,
                    source_type=st.session_state.source_type,
                )
                st.session_state.stage = "results"
            else:
                st.session_state.current_q += 1

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_results() -> None:
    answers = st.session_state.answers
    summary = compute_result_summary(answers)
    pct = summary["score_pct"]

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"""
            <div class="results-hero">
                <div class="section-title">Quiz Results</div>
                <div class="results-score">{pct}%</div>
                <div class="muted-text">
                    Student: {st.session_state.user_name} | Source: {st.session_state.source_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Create New Quiz", use_container_width=True):
            reset_quiz()
            st.rerun()

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="metric-chip">Score Percentage: {pct}%</div>
            <div class="metric-chip">Right Answers: {summary['correct']}</div>
            <div class="metric-chip">Wrong Answers: {summary['incorrect']}</div>
            <div class="metric-chip">Average Score: {summary['avg_score']:.1f}/10</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Feedback Summary")
    if pct >= 75:
        st.success("Strong performance. Your answers show solid understanding of the source material.")
    elif pct >= 40:
        st.warning("Decent attempt. Review the incorrect answers and feedback to improve your score.")
    else:
        st.error("This attempt needs revision. Focus on the feedback and the correct answers below.")

    for idx, answer in enumerate(answers, start=1):
        with st.expander(f"Q{idx}: {answer['question']}"):
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(f"**Question Type:** {answer['type']}")
            st.markdown(f"**Your Answer:** {answer['user_answer'] or '_No answer provided_'}")
            if answer["is_correct"]:
                st.markdown(
                    f"""
                    <div class="answer-status-box correct">
                        Correct Answer: {answer['correct_answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="answer-status-box wrong">
                        Wrong Answer. Correct Answer: {answer['correct_answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Score:** {answer['score']}/10")
            st.info(answer["feedback"])
            st.markdown("</div>", unsafe_allow_html=True)

    export_payload = {
        "student_name": st.session_state.user_name,
        "session_id": st.session_state.session_id,
        "source_name": st.session_state.source_name,
        "summary": summary,
        "answers": answers,
    }
    st.download_button(
        "Download Result JSON",
        data=json.dumps(export_payload, indent=2),
        file_name=f"quiz_result_{st.session_state.session_id}.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    ensure_directory(UPLOAD_DIR, "Upload directory")
    init_state()
    ollama_ok, _ = check_ollama()
    model = st.session_state.selected_model

    if st.session_state.stage == "setup":
        render_setup(ollama_ok=ollama_ok, model=model)
    elif st.session_state.stage == "quiz":
        render_quiz(model=model)
    else:
        render_results()


if __name__ == "__main__":
    main()
