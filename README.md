# Mindmaply - LeedsHack2026

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Google gemini API key (for core AI engine) - place this in a `.env` file in the `mindmap_chat` directory in the format GEMINI_API_KEY = "your_api_key_here"
```

### Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r my-fastapi-app/requirements.txt
```

## Running the App

### FastAPI App (Port 8000)

**Option 1: Using launch script (Windows)**
```powershell
.\launch-fastapi.ps1
```

**Option 2: Using launch script (macOS/Linux)**
```bash
chmod +x launch-fastapi.sh
./launch-fastapi.sh
```

**Option 3: Manual**
```bash
cd my-fastapi-app
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Access at: **http://127.0.0.1:8000**

API docs: **http://127.0.0.1:8000/docs**

## Project Structure

```text
LeedsHack2026/
├── mindmap_chat/           # Core AI Engine (Gemini Pro)
│   ├── core/               # Intent detection & graph logic
│   ├── llm/                # API clients & prompt templates
│   ├── utils/              # Helper functions
│   ├── config.py           # App configuration
│   ├── conversation.py     # Conversation orchestration	
│   ├── main.py             # CLI Entry point
│   ├── models.py           # Graph & Block data structures
│   ├── storage.py          # Persistence logic
│   └── requirements.txt    # Backend dependencies
├── my-fastapi-app/         # FastAPI Web Interface
│   ├── app/                # API endpoints & initialization
│   ├── static/             # CSS & D3.js visualization logic
│   ├── templates/          # HTML templates
│   └── requirements.txt    # Web dependencies
├── launch-fastapi.ps1      # Launcher (Windows)
├── launch-fastapi.sh       # Launcher (Bash)
└── README.md
```


