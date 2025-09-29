# Enhanced Semantic RAG for Wroclaw University Documents

Semantic chunking, indexing, and search over university PDFs with:
- Topic‑aware chunking
- SentenceTransformer embeddings
- FAISS similarity search
- Optional Streamlit UI
- Basic CLI semantic search

## Project Structure
- `documents/`: Source PDFs
- `semantic_chunking.py`: Build enhanced KB, Streamlit app, RAG chat
- `semantic_search.py`: Simple CLI semantic search over `semantic_knowledge_base.json`
- `requirements.txt`: Base dependencies
- `semantic_knowledge_base.json`: Legacy/simple KB (if already generated)

## Prerequisites
- Python 3.9+ recommended
- macOS: `brew install cmake` may help with some builds
- Ensure you have enough RAM/disk for embedding PDFs

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Extra deps used by semantic_chunking.py:
pip install pandas streamlit faiss-cpu openai
# NLTK data (automatic in code; manual fallback):
python -c "import nltk; nltk.download('punkt')"
```

## Data
Place PDFs in:
- `polishh/documents/`

The repo already contains several sample PDFs in that folder.

## Option A: Enhanced KB + Streamlit App
Build an enhanced knowledge base from PDFs and explore via UI.

1) Run the app:
```bash
cd polishh
streamlit run semantic_chunking.py
```

2) In the sidebar:
- Choose an embedding model (default `paraphrase-multilingual-MiniLM-L12-v2`)
- Adjust chunking thresholds if needed
- Upload PDFs or use the existing `documents/` by uploading them in the UI
- Click “Build Enhanced KB”
- Optionally “Save KB” / “Load KB”

3) Chat
- Enter a question
- Optionally filter sources and set result count
- Inspect sources, similarity scores, clusters, and metadata

Optional OpenAI responses:
```bash
export OPENAI_API_KEY=your_key_here
# The app will use it for LLM responses; otherwise a fallback response is shown.
```

Artifacts saved by the enhanced KB (when you click “Save KB”):
- `enhanced_university_kb.index`
- `enhanced_university_kb.enhanced`

## Option B: Simple CLI Semantic Search
Search over a prebuilt `semantic_knowledge_base.json` (legacy/simple format).

1) Ensure the file exists (already in repo or created separately).
2) Run:
```bash
cd polishh
python semantic_search.py
```
Then follow the interactive prompts:
- Type your question
- Use `help`, `config`, `stats`
- Type `exit`/`quit` to leave

Note: This path does not use FAISS; it searches the JSON KB directly.

## Troubleshooting
- Model download issues: ensure internet access; try `pip install --upgrade sentence-transformers`.
- NLTK SSL issues: the code sets an unverified HTTPS context; if it still fails, manually run `nltk.download('punkt')`.
- FAISS install problems: use `faiss-cpu` (already noted) and a recent pip.
- Missing deps: install extras: `pip install pandas streamlit faiss-cpu openai`.

## License
For academic/educational use at Wroclaw University of Economics and Business. Adapt as needed.
