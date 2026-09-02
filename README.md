# VietLink RAG

VietLink RAG turns an initial request into a clearer, implementation-ready request. It retrieves relevant knowledge-base (KB) entries, asks focused clarification questions, and returns the original request with ambiguity removed while preserving its structure.

## Features

- Select GPT, Claude, or Gemini as the clarification provider.
- Analyze uploaded or pasted conversations and extract reusable ambiguity KB entries.
- Retrieve provider-specific KB entries from `data/<provider>/kbs.json`.
- Use direct, sentence-level, or normalized retrieval based on input length.
- Ask a configurable number of clarification questions.
- Preserve the original request's headings, bullets, paragraph order, and language in the final clarified request.

## Prerequisites

- Git
- Python 3.10 or later
- An API key for at least one supported provider:
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Google Gemini: `GOOGLE_API_KEY`

## Clone and Run Locally (Windows, macOS, and Linux)

1. Clone the repository and enter the project directory.

   **Windows PowerShell**

   ```powershell
   git clone https://github.com/baoduongphu/vietlink_chatbot.git
   cd vietlink_chatbot
   ```

   **macOS or Linux**

   ```bash
   git clone https://github.com/baoduongphu/vietlink_chatbot.git
   cd vietlink_chatbot
   ```

2. Create a virtual environment and install dependencies.

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   **macOS or Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Create the local secrets file.

   **Windows PowerShell**

   ```powershell
   Copy-Item .env.example .env
   ```

   **macOS or Linux**

   ```bash
   cp .env.example .env
   ```

4. Open `.env` and add the API key for every provider you want to use. Only configure keys you intend to use.

   ```env
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   ```

   Optional model overrides are documented in `.env.example`.

   The analysis extraction service uses separate model variables for its two
   stages:

   ```env
   OPENAI_ANALYSIS_MODEL=gpt-5
   OPENAI_EXTRACTION_MODEL=gpt-5-mini
   ANTHROPIC_ANALYSIS_MODEL=claude-opus-4-8
   ANTHROPIC_EXTRACTION_MODEL=claude-haiku-4-5-20251001
   GOOGLE_ANALYSIS_MODEL=gemini-pro-latest
   GOOGLE_EXTRACTION_MODEL=gemini-flash-lite-latest
   ```

5. Verify that the KB files exist and are valid JSON arrays:

   ```text
   data/gpt/kbs.json
   data/claude/kbs.json
   data/gemini/kbs.json
   ```

6. Start Streamlit.

   **Windows PowerShell**

   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

   **macOS or Linux**

   ```bash
   python -m streamlit run app.py
   ```

   Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The sidebar's **Service** selector exposes both workflows:

- **Requirement clarification** runs the existing RAG clarification chatbot.
- **Analysis extraction** runs the two-stage analysis/extraction service, displays
  the analysis report and KB entries, and provides Markdown and JSON downloads.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub and create a Streamlit Community Cloud app.
2. Set the entrypoint to `app.py`.
3. Add at least one provider key under the app's **Secrets** settings:

   ```toml
   OPENAI_API_KEY = "..."
   ANTHROPIC_API_KEY = "..."
   GOOGLE_API_KEY = "..."

   # Optional analysis extraction overrides
   OPENAI_ANALYSIS_MODEL = "gpt-5"
   OPENAI_EXTRACTION_MODEL = "gpt-5-mini"
   ```

4. Deploy the app. Both chatbot services are available from the sidebar.

## Workflow

1. Inputs of 40 tokens or fewer use direct retrieval.
2. Inputs from 41 to 150 tokens use sentence-level retrieval.
3. Longer inputs are normalized with `NORMALIZATION_PROMPT_TEMPLATE`, then retrieved using the core request.
4. `CLARIFICATION_PROMPT_TEMPLATE` asks one question per turn or produces the final clarified request.
5. The final request keeps the original format and adds clarification details only where they are relevant.

## Chatbot Settings

The sidebar exposes these runtime settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| Candidate KB entries | 5 | Maximum retrieved KB entries used for clarification. |
| Similarity threshold | 0.0 | Minimum similarity score for a KB entry to be included. |
| Maximum output tokens | 8192 | Provider output token limit. |
| Maximum clarification rounds | 3 | Maximum questions asked before finalization. |

Changing a setting starts a new conversation session to prevent mixed configurations.
