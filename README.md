# EduSafeAI Hub

A Flask-based AI toolkit for K-12 educators worldwide — with lesson planning, feedback, differentiation, policy drafting, family communication, and assessment support.

## Highlights
- Educator-first UI with focused tools (lesson, IEP/ELL, policy, feedback, unit planning, quiz/rubric, and response refinement)
- **Start here** quick launcher with expanded goals (including quiz, rubric, and response refinement)
- Visible PII safety reminder in the UI (teacher-use guardrail)
- **Multi-AI powered**: automatically tries 7+ free-tier LLM providers in order for maximum availability
- Standards-aligned planning supporting Common Core, NGSS, WIDA, IB, and local standards

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:5000`.

## Environment variables
### Shared
- `PORT` (optional): server port, defaults to `5000`

### AI Provider API Keys (add as many as possible for best reliability)

The app tries each provider in order and falls back gracefully.

- `GROQ_KEY`: Groq API key (https://console.groq.com)
- `CEREBRAS_KEY`: Cerebras API key (https://cloud.cerebras.ai)
- `GEMINI_KEY`: Google AI Studio API key (https://aistudio.google.com)
- `COHERE_KEY`: Cohere API key (https://dashboard.cohere.com)
- `MISTRAL_KEY`: Mistral AI API key (https://console.mistral.ai)
- `OPENROUTER_KEY`: OpenRouter API key (https://openrouter.ai)
- `HF_KEY`: Hugging Face API token (https://huggingface.co/settings/tokens)

## Notes
- If you see "All AI providers are currently busy", verify that at least one provider API key is set, then retry.
- The app performs light request validation and returns structured `400` errors for missing required fields.
- Responses for identical requests are cached in-memory for up to 1 hour (max 500 entries), reducing API usage for common educator queries.