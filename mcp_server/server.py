# server.py — ultra-simple MCP server for debugging
import os, sys, asyncio, random
from pathlib import Path
from typing import Optional, Dict
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List
import typing
import ast
from pathlib import Path
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import json
from typing import Optional
try:
    from typing import Annotated 
except ImportError:
    from typing_extensions import Annotated
setattr(typing, "Annotated", Annotated)
print("[DEBUG] Annotated patched into typing:", hasattr(typing, "Annotated"))
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from sentence_transformers import util
DOC_PATH = Path("docs/knowledge.pdf")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(path: Path, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Read PDF and split into overlapping chunks."""
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    # Chunk with overlap
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

load_dotenv()
from sentence_transformers import util




def build_index(chunks: list[str]):
    embeddings = EMBED_MODEL.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_chunks(query: str, index, chunks, k: int = 5) -> str:
    query_emb = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return "\n".join(chunks[i] for i in I[0])


def grounding_score(generated: str, context: str) -> float:
    emb_gen = EMBED_MODEL.encode([generated], convert_to_tensor=True)
    emb_ctx = EMBED_MODEL.encode([context], convert_to_tensor=True)
    sim = util.cos_sim(emb_gen, emb_ctx).item()
    return round(sim * 100, 2)


API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env")


if not DOC_PATH.exists():
    raise FileNotFoundError(f"Missing {DOC_PATH.resolve()}")


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY, temperature=0.2)
db = SQLDatabase.from_uri("sqlite:///students.db")
query_tool = QuerySQLDatabaseTool(db=db)

EXAM_HALL_SCHEDULE: Dict[str, Dict[str, str]] = {}

def generate_examhalls_schedule():
    """
    Generates an initial exam hall booking schedule for the next 30 days.
    - Each day has slots from 08:00 to 18:00 (hourly).
    
    """
    global EXAM_HALL_SCHEDULE
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 19)]  # 8 AM to 6 PM

    for i in range(30):  # Next 30 days
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        EXAM_HALL_SCHEDULE[date_str] = {time: "available" for time in possible_times}

# Initialize the schedule on module load
generate_examhalls_schedule()

schema_hint = """
You are an expert SQL assistant. The database schema is:

TABLE students(id INTEGER PK, name TEXT UNIQUE)
TABLE timeslots(id INTEGER PK, day TEXT, start TEXT, "end" TEXT)
TABLE availability(student_id INTEGER, timeslot_id INTEGER, is_free INTEGER)

Rules:
- SQLite syntax only.
- "is_free = 1" means student is free.
- Always join availability → students → timeslots.
"""

#  MCP app 
mcp = FastMCP(name="simple_quiz_tools", host="0.0.0.0", port="8787")



@mcp.tool()
def generate_exam_with_plan_and_rag(plan_json: str, topic: Optional[str] = None) -> str:
    """
    Generate an exam strictly based on:
    1. The exam plan JSON from Examplanner Agent (blueprint: structure, difficulty, focus topics).
    2. Retrieved course text from PDF chapters (RAG).
    """

    # Step A: Load + index course text
    chunks = load_pdf(DOC_PATH)
    index, chunk_list = build_index(chunks)

    # Step B: Retrieve relevant chunks based on exam plan (and topic if provided)
    query = topic or "general exam content"
    rag_context = retrieve_chunks(query, index, chunk_list, k=8)  
    print("=== RETRIEVED CONTEXT ===")
    print(rag_context)

    # Step C: Build prompt
    topic_line = f"Topic: {topic}\n" if topic else ""
    prompt = f"""
You are ExamGenerator, the Exam Generator Agent.

Use BOTH the exam plan (blueprint) and the retrieved study material below.

--- Exam Plan ---
{plan_json}

--- Study Material (retrieved RAG chunks) ---
{rag_context}

Instructions:
- Follow the plan strictly for number of questions, structure, and difficulty distribution.
- You MUST use the retrieved study material for all definitions, facts, and MCQs. 
Do not invent content. 
When possible, copy phrases directly from the study material into questions and answers.
- For EACH question, prepend a label indicating:
    [Topic: <focus_topic> | Weight: <percentage>%]
- Generate exactly 10 questions:
  * 6 Multiple Choice (MCQ, A–D)
  * 2 True/False
  * 2 Short Answer (1–2 sentences).
- After listing all questions, include an **Answer Key** mapping Q# to the correct answer.
""".strip()

    #  Call LLM
    result = llm.invoke(prompt)
    text = getattr(result, "content", None) or getattr(result, "text", None) or ""
    if isinstance(text, list):
        text = "".join(getattr(p, "text", "") for p in text if getattr(p, "text", None))

    generated_exam = text.strip() or "No output produced."

    
    grounding = grounding_score(generated_exam, rag_context)
    generated_exam += f"\n\n---\nGrounding Score: {grounding}%"

    return generated_exam

@mcp.tool()
def list_examhall_availabilities(date: str) -> str:
    """
    Lists the available and booked slots for the exam hall on a given date.
    
    Args:
        date: Date in YYYY-MM-DD format.

    Returns:
        A string showing available and booked slots for the exam hall.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD."

    daily_schedule = EXAM_HALL_SCHEDULE.get(date)
    if not daily_schedule:
        return f"No exam hall schedule found for {date}."

    available_slots = [
        time for time, status in daily_schedule.items() if status == "available"
    ]
    booked_slots = {
        time: booked_by
        for time, booked_by in daily_schedule.items()
        if booked_by != "available"
    }

    result = f"Exam hall schedule for {date}:\n"
    result += f"Available slots: {', '.join(available_slots) if available_slots else 'None'}\n"
    
    if booked_slots:
        result += "Booked slots:\n"
        for time, booked_by in booked_slots.items():
            result += f"  {time}: {booked_by}\n"
    else:
        result += "Booked slots: None\n"
    
    return result

@mcp.tool()
def  book_examhall(
    date: str,
    student_start: str,
    student_end: str,
    reservation_name: str
) -> str:
    """
    Book an exam hall if both:
      1. All students are available during the requested hours.
      2. The exam hall is also free in the same slots.

    Args:
        date: Date in YYYY-MM-DD format.
        student_start: Start time (from student availability) in HH:MM format.
        student_end: End time (from student availability) in HH:MM format.
        reservation_name: Name or event for the booking.

    Returns:
        Confirmation string or error message.
    """
    try:
        start_dt = datetime.strptime(f"{date} {student_start}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {student_end}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: Invalid date or time format. Use YYYY-MM-DD and HH:MM."

    if start_dt >= end_dt:
        return "Error: Start time must be before end time."

    if date not in EXAM_HALL_SCHEDULE:
        return f"Error: No exam hall schedule exists for {date}."

    if not reservation_name.strip():
        return "Error: Cannot book without a reservation name."

    # Required hourly slots
    required_slots = []
    current_time = start_dt
    while current_time < end_dt:
        required_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(hours=1)

    daily_schedule = EXAM_HALL_SCHEDULE.get(date, {})

    # Check if ALL required slots are available
    unavailable = []
    for slot in required_slots:
        if daily_schedule.get(slot, "booked") != "available":
            unavailable.append((slot, daily_schedule.get(slot, "unknown")))

    if unavailable:
        # At least one slot is not free
        conflicts = "; ".join([f"{s} (booked by {b})" for s, b in unavailable])
        return f"Sorry, the exam hall is not free during the required slots: {conflicts}."

    # Book the hall
    for slot in required_slots:
        EXAM_HALL_SCHEDULE[date][slot] = reservation_name

    return f" Success: Exam hall booked for {reservation_name} on {date} from {student_start} to {student_end}."

@mcp.tool()
def get_exam_hall_summary() -> str:
    """
    Provides a summary of exam hall bookings for the next 7 days.
    
    Returns:
        A string summary of upcoming bookings.
    """
    today = date.today()
    summary = "Exam Hall Booking Summary (Next 7 Days):\n\n"
    
    for i in range(7):
        check_date = today + timedelta(days=i)
        date_str = check_date.strftime("%Y-%m-%d")
        day_name = check_date.strftime("%A")
        
        daily_schedule = EXAM_HALL_SCHEDULE.get(date_str, {})
        booked_slots = {
            time: booked_by
            for time, booked_by in daily_schedule.items()
            if booked_by != "available"
        }
        
        summary += f"{day_name}, {date_str}:\n"
        if booked_slots:
            for time, booked_by in sorted(booked_slots.items()):
                summary += f"  {time}: {booked_by}\n"
        else:
            summary += "  No bookings\n"
        summary += "\n"
    
    return summary
@mcp.tool()
def generate_exam_plan(topic: str) -> str:
    """
    Generates an exam plan including difficulty, coefficients, and focus topics
    for the given subject. Return a string for the user.
    Args:
        topic: The exam topic, e.g., "Physics", "Machine learning", "Maths".
    """
    import random, json

    difficulty_levels = ["Easy", "Medium", "Hard"]
    difficulty = random.choice(difficulty_levels)

    possible_subtopics = [
        "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
        "Semi-Supervised Learning", "Self-Supervised Learning",
        "Linear Algebra", "Calculus for ML", "Probability Theory", "Statistics",
        "Optimization Techniques", "Information Theory",
        "Linear Regression", "Logistic Regression", "Decision Trees", "Random Forests",
        "Support Vector Machines", "Naive Bayes", "K-Nearest Neighbors",
        "K-Means Clustering", "PCA", "LDA",
        "Perceptron", "Feedforward Neural Networks", "Backpropagation", "CNNs",
        "RNNs", "LSTMs", "GRUs", "Transformers", "Attention Mechanisms",
        "Autoencoders", "GANs",
        "Natural Language Processing", "Computer Vision", "Speech Recognition",
        "Time Series Forecasting", "Anomaly Detection", "Recommendation Systems",
        "Feature Engineering", "Feature Scaling", "Data Preprocessing",
        "Model Evaluation Metrics", "Cross-Validation", "Hyperparameter Tuning",
        "Ensemble Methods", "Bagging", "Boosting", "Stacking",
        "Model Deployment", "MLOps", "Explainable AI", "Bias and Fairness in ML",
        "Model Interpretability", "Ethical AI"
    ]

    focus_topics = random.sample(possible_subtopics, k=random.randint(4, 6))
    random_weights = [random.random() for _ in focus_topics]
    total = sum(random_weights)
    coefficients = {
        focus_topics[i]: round(random_weights[i] / total, 2) for i in range(len(focus_topics))
    }

    plan = {
        "topic": topic,
        "difficulty": difficulty,
        "coefficients": coefficients,
        "focus_topics": focus_topics
    }

    return json.dumps(plan, indent=2)


@mcp.tool()
def query_students_availability() -> dict:
    """Always queries: 'When are all 10 students free together?'
    Returns structured data (not full English explanation)."""

    question = "When are all 10 students free together?"

    # Step A: Generate SQL
    sql_prompt = f"""{schema_hint}

Question: {question}
Write a valid SQLite query only.
IMPORTANT: Do NOT include markdown code fences or labels like ```sql.
Just output pure SQL.
"""
    sql = llm.invoke(sql_prompt).content.strip()
    if sql.startswith("```"):
        sql = sql.strip("`").replace("sqlite", "").replace("sql", "").strip()

    # Step B: Run SQL
    result = query_tool.run(sql)

    
    # result will look like: [('Fri', '11:00', '12:00'), ('Mon', '11:00', '12:00'), ...]
    free_slots = []
    try:
    # Parse string into real Python list
     parsed = ast.literal_eval(result) if isinstance(result, str) else result

     if isinstance(parsed, (list, tuple)):
        for row in parsed:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                free_slots.append({
                    "day": row[0],
                    "start": row[1],
                    "end": row[2]
                })
    except Exception as e:
     print("Error parsing result:", e)

    return {"free_slots": free_slots}
if __name__ == "__main__":

    test_plan = {
        "topic": "Machine Learning",
        "difficulty": "Medium",
        "coefficients": {"Supervised Learning": 0.5, "Optimization": 0.5},
        "focus_topics": ["Supervised Learning", "Optimization"]
    }
    
    out = generate_exam_with_plan_and_rag(json.dumps(test_plan), topic="Machine Learning")
    print("=== GENERATED EXAM ===")
    print(out)

    mcp.run(transport="sse")