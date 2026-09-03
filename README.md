# VietLink RAG

VietLink RAG turns an initial request into a clearer, implementation-ready request. It retrieves relevant knowledge-base (KB) entries, asks focused clarification questions, and returns the original request with ambiguity removed while preserving its structure.

## Architecture and Technical Design

VietLink RAG is a **retrieval-augmented generation (RAG)** system designed specifically for requirements clarification. Unlike a conventional document question-answering chatbot, it does not use retrieved content to answer the request directly. Each knowledge-base entry describes a common type of missing or ambiguous information. Relevant entries are added to the prompt so the LLM can determine what additional information to ask for and when the request is sufficiently clear for implementation.

```text
Initial request
      |
      v
Select a retrieval strategy based on input length
      |
      +-- <= 40 words: embed the complete request
      +-- 41-150 words: split into sentences, embed each, merge results
      +-- > 150 words: normalize with a lightweight LLM, embed core_request
      |
      v
SentenceTransformer -> ChromaDB (cosine similarity)
      |
      v
Top-K KB entries + original request + clarification history
      |
      v
LLM produces JSON: ask the next question or return the clarified request
```

### Core components

| Component | File | Responsibility |
| --- | --- | --- |
| UI and application state | `app.py` | Provides the Streamlit interface, accepts runtime configuration, and stores the `ClarificationSession` in `st.session_state`. |
| Configuration | `rag/config.py` | Defines providers, generation and lightweight models, KB/index paths, and runtime parameter validation. |
| Embedding | `rag/embedding.py` | Loads a fixed multilingual embedding model and produces normalized vectors. |
| Vector store | `rag/vector_db.py` | Loads and validates the KB, builds a provider-specific Chroma collection, and performs cosine-similarity searches. |
| Retriever | `rag/retriever.py` | Selects one of three retrieval strategies, applies the similarity threshold, and returns up to `top_k` candidates. |
| RAG orchestration | `rag/rag_system.py` | Initializes dependencies, starts sessions, manages clarification turns, and finalizes requests. |
| LLM integration | `rag/client_manager.py`, `rag/llm.py` | Lazily initializes provider clients, calls GPT/Claude/Gemini APIs, and normalizes JSON responses. |
| Prompts | `rag/prompts.py` | Defines the contracts for request normalization and requirements clarification. |

### Knowledge base and vector index construction

Each provider has an independent KB at `data/<provider>/kbs.json` and a persistent collection under `.chroma/<provider>`. When `RAGSystem` is initialized, the system performs the following steps:

1. Read the JSON file and verify that it contains a non-empty array.
2. Verify that every entry contains at least `knowledge_id` and `trigger`.
3. Delete the previous collection and recreate it with `hnsw:space = cosine`.
4. Embed the `trigger` field of every KB entry using `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
5. Store each vector with its `knowledge_id` as the document ID. The complete entry remains in memory so it can be added to the prompt after retrieval.

Embeddings are normalized with `normalize_embeddings=True`. Chroma returns cosine distance, so the displayed and filtered score is calculated as `similarity = 1 - distance`. `similarity_threshold` removes low-scoring results, while `top_k` limits the final number of candidates. The index is currently rebuilt whenever a new `RAGSystem` instance is created; incremental updates and KB version checks are not implemented.

A KB entry may also contain `title`, `category`, `description`, `detection_rules`, `clarification_questions`, `recommendation`, `expected_outcome`, and `source`. Retrieval searches only the `trigger` vectors, but the complete matching entries are passed to the LLM so it can evaluate their applicability and formulate a clarification question.

### Adaptive retrieval pipeline

Input length is estimated using the number of elements returned by `text.split()`, rather than a model-specific tokenizer.

- **Direct (`<= 40`)**: embeds the complete request and performs one vector search.
- **Sentence chunk (`41-150`)**: splits the input on `.`, `!`, `?`, or line breaks; searches with each sentence; deduplicates matches by `knowledge_id`; retains the highest score for each entry; and selects the global top-K results.
- **Normalized (`> 150`)**: calls the provider's lightweight model with temperature `0` and a `600`-token output limit to extract `core_request`, `stated_info`, and `background_context`. Only `core_request` is used for vector retrieval. The original request is used as a fallback when `core_request` is empty.

Normalization does not replace the original content stored in the session. The original request remains the source input for clarification, reducing the risk of losing structure or details when the retrieval query is shortened.

### Clarification and generation loop

After retrieval, `RAGSystem` creates a `ClarificationSession` containing the original request, candidate KB entries, retrieval strategy, clarification turns, and final result. At every turn, the prompt sent to the generation model contains:

- the original request without modification;
- the complete question-and-answer history as JSON;
- the complete contents of the retrieved KB entries;
- the current round number and maximum round limit.

The LLM must return a JSON object containing `status`, `matches`, `next_question`, `clarified_request`, and `assumptions_made`. For `status="ask"`, the system accepts one question only when the round limit has not been reached. In every other case, the session is finalized. If the model returns no `clarified_request`, the system falls back to the original request. At the round limit, the prompt instructs the model to apply reasonable assumptions and report them through `assumptions_made`.

Conversation history is explicitly included in every prompt; provider SDKs do not maintain a long-lived chat session. This produces consistent behavior across OpenAI, Anthropic, and Gemini. The prompt also requires the model to preserve the input language, voice, headings, paragraph order, and lists, making only the smallest changes needed to remove ambiguity.

### Providers and resource lifecycle

Each provider uses two model roles: `generation_model` for the clarification loop and `light_model` for normalizing long inputs. Model names can be overridden through environment variables. API clients are initialized lazily and cached by `ClientManager`. Streamlit caches the `RAGSystem` instance, embedding model, and vector index using `(provider, top_k, similarity_threshold, max_tokens, max_question_rounds)` as the configuration key.

When the configuration changes, the UI discards the current conversation session to prevent history from being mixed across configurations. API keys are read only from environment variables or Streamlit secrets and are not stored in ChromaDB or the session object.

### Current limitations

- There is no reranker or hybrid search; relevance is based only on cosine similarity between `trigger` vectors.
- The final result does not include source citations; the KB is used as guidance for detecting missing information.
- Output parsing depends on model-generated JSON. A response without a valid JSON object raises `ConfigurationError`.
- ChromaDB persists data on disk, but the collection is still deleted and rebuilt whenever the corresponding system is initialized.
- Conversation state exists only in the Streamlit session. It is not stored in a database or shared between users or browser sessions.

## Features

- Select GPT, Claude, or Gemini as the clarification provider.
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
