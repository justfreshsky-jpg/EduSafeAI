# EduSafeAI Hub

A Flask-based AI toolkit for K-12 educators across the United States — with lesson planning, feedback, differentiation, policy drafting, family communication, and assessment support tailored to every US state's standards.

## Highlights
- **USA State Selector**: Choose Federal (CCSS/NAEP) or any of the 50 states — all content, standards codes, and assessment names update dynamically
- Educator-first UI with focused tools (lesson, IEP/ELL, policy, feedback, unit planning, quiz/rubric, and response refinement)
- **Standards-aligned for every state**: NJ (NJSLS/NJSLA), TX (TEKS/STAAR), FL (B.E.S.T./F.A.S.T.), CA (CCSS/CAASPP), NY (Next Gen/Regents), GA (GSE/Milestones), VA (SOL), PA (PA Core/PSSA), and all 50 states + Federal
- Visible PII safety reminder in the UI (teacher-use guardrail)
- **Multi-AI powered**: automatically tries 7+ free-tier LLM providers in order for maximum availability
- State-specific system prompts, fallback content, and assessment question generation

## State Selection Feature

The state selector dropdown at the top of the page lets educators instantly switch between:
- 🇺🇸 **Federal (USA)** — Common Core, NGSS, ESSA, NAEP
- **All 50 US states** — each with their own standards body, assessment system, and standard code formats

When a state is selected, the following update automatically:
- Assessment tab label (e.g., "NJSLA Prep", "STAAR Prep", "F.A.S.T. Prep")
- Assessment practice question generator uses state-specific system prompt
- Lesson designer standards dropdown shows state-specific standard codes
- Science subject label reflects state's science framework
- All AI responses reference the selected state's standards and assessments

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