## Quiz App

This project is a Streamlit-based quiz generator and evaluator built for the internship training project.
It uses Ollama as the local LLM backend and can generate quizzes from:

- PDF files
- TXT files
- DOCX files
- CSV files
- A pasted paragraph

The app supports:

- Student name registration
- Multiple Choice and Short Answer questions
- Answer evaluation with feedback
- Final score percentage
- Right and wrong answer summary
- JSON export of quiz results

## Project Structure

- [app.py](C:/Users/Dell/Desktop/quiz_app/app.py) - Streamlit UI
- [modules/ingestion.py](C:/Users/Dell/Desktop/quiz_app/modules/ingestion.py) - file and text ingestion
- [modules/question_gen.py](C:/Users/Dell/Desktop/quiz_app/modules/question_gen.py) - quiz question generation
- [modules/evaluator.py](C:/Users/Dell/Desktop/quiz_app/modules/evaluator.py) - answer evaluation
- [modules/session.py](C:/Users/Dell/Desktop/quiz_app/modules/session.py) - session save and score summary

## Setup

1. Install Python 3.10 or newer.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama:

```bash
ollama serve
```

4. Pull a model if needed:

```bash
ollama pull llama3.1
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

## Workflow

1. Enter the student's name.
2. Upload a document or paste a paragraph.
3. Choose the number of questions, difficulty, and question types.
4. Generate the quiz using Ollama.
5. Answer each question.
6. Review the final result summary with percentage, correct answers, wrong answers, and feedback.

## Notes

- Ollama must be running locally before generating questions or evaluating short answers.
- Multiple Choice questions are evaluated instantly.
- Short Answer questions are evaluated semantically using the selected Ollama model.
