import os

OPENAI_API_KEY_PATH = os.path.join(os.environ["HOME"], ".keys", "openai_api_key.txt")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY and os.path.exists(OPENAI_API_KEY_PATH):
    with open(OPENAI_API_KEY_PATH) as f:
        OPENAI_API_KEY = f.read().strip()

JWT_SECRET_KEY = "replace_me"

#### Prompts - override to set

CONDENSE_QUESTION_PROMPT = None
QA_PROMPT = None

#### Document(s) to load

AI_REPORT_PDF_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../examples", "example.pdf"
)

AI_REPORT_CHROMADB_DIRECTORY = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../data", "ai_report_chromadb"
)
