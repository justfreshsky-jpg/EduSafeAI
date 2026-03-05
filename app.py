import collections
import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
os.environ.setdefault('HTTPX_PROXIES', 'null')  # Fix Render/httpx proxies bug
import requests
import threading
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

# ── LOGGING ──────────────────────────────────────────────────
_log_fmt = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
_log_dir = os.environ.get('LOG_DIR')
if _log_dir:
    _file_handler = RotatingFileHandler(
        os.path.join(_log_dir, 'edusafeai.log'), maxBytes=5_000_000, backupCount=3
    )
    _file_handler.setFormatter(_log_fmt)
    logging.getLogger().addHandler(_file_handler)

app = Flask(__name__)

GROQ_KEY = os.environ.get('GROQ_KEY')
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# ── BLOG SCRAPER (background, server-side only) ──────────────
_cache = {"content": "", "last": 0}
_blog_lock = threading.Lock()

FALLBACK = """
[LESSON] Effective lessons require hooks, differentiation, formative checks, and exit tickets.
[IEP/ELL] ELL Proficiency levels: 1=Entering, 2=Emerging, 3=Developing, 4=Expanding, 5=Bridging (based on WIDA framework).
[STANDARDS] Common frameworks: Common Core (US), National Curriculum (UK), IB, Cambridge, state/provincial standards.
[POLICY] AI use policies must address plagiarism, data privacy, student safety, and academic integrity.
[504/IEP] 504 covers access accommodations; IEP covers specialized instruction under IDEA (US) or equivalent local law.
[EMAIL] Parent communications should be culturally responsive and available in families' home languages.
[FEEDBACK] Rubric-aligned feedback: Strength, Evidence, Reasoning, Next Step.
[UNIT] 2-week unit: overview, daily breakdown, formative + summative assessments.
"""

def _fetch_blog():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        sources = [
            "https://www.ed.gov/",
            "https://www.understood.org/",
        ]
        combined = ""
        for url in sources:
            try:
                r = requests.get(url, headers=headers, timeout=6)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                combined += text[:1500] + "\n---\n"
            except Exception:
                app.logger.warning("Failed to fetch blog content from %s", url, exc_info=True)
                continue
        if combined.strip():
            with _blog_lock:
                _cache["content"] = combined[:5000]
                _cache["last"] = time.time()
    except Exception:
        app.logger.exception("Blog content refresh failed")

def _bg_refresh():
    time.sleep(5)  # initial delay to avoid crash on import
    while True:
        _fetch_blog()
        time.sleep(3600)

threading.Thread(target=_bg_refresh, daemon=True).start()

def get_context():
    with _blog_lock:
        return _cache["content"] if _cache["content"] else FALLBACK

# ── LLM ─────────────────────────────────────────────────────
def _focus_prompt(focus):
    return (
        "You are supporting K-12 educators across all regions and countries.\n"
        "Provide practical, classroom-ready guidance aligned to widely-used standards "
        "(Common Core, NGSS, WIDA, IB, or the educator's local standards).\n"
        "Include culturally responsive and inclusive practices.\n"
        "Use clear steps, sample phrasing, and quick-use templates."
    )


def _llm_via_groq(full_system, user):
    if not client:
        raise ValueError("GROQ_KEY not configured")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": full_system}, {"role": "user", "content": user}],
        max_tokens=1500,
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").replace('**', '').strip()


def _llm_via_cerebras(full_system, user):
    key = os.environ.get('CEREBRAS_KEY')
    if not key:
        raise ValueError("CEREBRAS_KEY not configured")
    resp = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_gemini(full_system, user):
    key = os.environ.get('GEMINI_KEY')
    if not key:
        raise ValueError("GEMINI_KEY not configured")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": full_system + "\n\n" + user}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1500},
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return (text or "").replace('**', '').strip()


def _llm_via_cohere(full_system, user):
    key = os.environ.get('COHERE_KEY')
    if not key:
        raise ValueError("COHERE_KEY not configured")
    resp = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "command-r-plus",
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1500,
            "temperature": 0.6
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("message", {}).get("content", [{}])[0].get("text", "")
    return (text or "").replace('**', '').strip()
    

def _llm_via_mistral(full_system, user):
    key = os.environ.get('MISTRAL_KEY')
    if not key:
        raise ValueError("MISTRAL_KEY not configured")
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_openrouter(full_system, user):
    key = os.environ.get('OPENROUTER_KEY')
    if not key:
        raise ValueError("OPENROUTER_KEY not configured")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://edusafeai.onrender.com",
            "X-Title": "EduSafeAI Hub",
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_huggingface(full_system, user):
    key = os.environ.get('HF_KEY')
    if not key:
        raise ValueError("HF_KEY not configured")
    resp = requests.post(
        "https://router.hugging-face.cn/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


# ── ORDERED FALLBACK DISPATCH ────────────────────────────────
_PROVIDERS = [
    ("groq", _llm_via_groq),
    ("cerebras", _llm_via_cerebras),
    ("gemini", _llm_via_gemini),
    ("cohere", _llm_via_cohere),
    ("mistral", _llm_via_mistral),
    ("openrouter", _llm_via_openrouter),
    ("huggingface", _llm_via_huggingface),
]

# ── RESPONSE CACHE ───────────────────────────────────────────
_resp_cache: collections.OrderedDict = collections.OrderedDict()
_CACHE_MAX = 500
_CACHE_TTL = 3600
_cache_lock = threading.Lock()

# ── RATE LIMITING ────────────────────────────────────────────
_RATE_LIMIT = 20       # requests per window per IP
_RATE_WINDOW = 60      # seconds
_rate_data: dict = {}
_rate_lock = threading.Lock()


def _check_rate_limit():
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or '0.0.0.0'
    now = time.time()
    with _rate_lock:
        if ip not in _rate_data:
            _rate_data[ip] = collections.deque()
        dq = _rate_data[ip]
        while dq and dq[0] < now - _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return False
        dq.append(now)
        # Clean up stale entries to prevent unbounded memory growth
        stale = [k for k, v in _rate_data.items() if not v]
        for k in stale:
            del _rate_data[k]
    return True


@app.before_request
def enforce_rate_limit():
    if request.method == 'POST':
        if not _check_rate_limit():
            return jsonify(error="Rate limit exceeded. Please wait a minute before making another request."), 429


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


def _cache_get(key):
    with _cache_lock:
        if key in _resp_cache:
            val, ts = _resp_cache[key]
            if time.time() - ts < _CACHE_TTL:
                _resp_cache.move_to_end(key)
                return val
            del _resp_cache[key]
    return None


def _cache_set(key, val):
    with _cache_lock:
        if key in _resp_cache:
            _resp_cache.move_to_end(key)
        _resp_cache[key] = (val, time.time())
        while len(_resp_cache) > _CACHE_MAX:
            _resp_cache.popitem(last=False)


def llm(system, user, focus="general"):
    focus_prompt = _focus_prompt(focus)
    full_system = (
    system
    + "\n\n"
    + focus_prompt
    + "\n\nOUTPUT FORMAT RULES:\n"
    + "• Always respond with complete, well-structured output.\n"
    + "• Use clear section headers (e.g., using ALL CAPS or bold labels).\n"
    + "• Each section should contain 2-4 concise, actionable sentences or bullet points.\n"
    + "• Never truncate or cut off mid-sentence or mid-section.\n"
    + "• Do not repeat the user's question back to them.\n"
    + "• Do not add meta-commentary like \"Here is your lesson plan:\" — just output the content directly.\n\n"
    + "Reference context:\n"
    + get_context()
)

    cache_key = hashlib.sha256((system + user + focus).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        app.logger.info("LLM request served from cache")
        return cached

    for name, func in _PROVIDERS:
        try:
            result = func(full_system, user)
            if result:
                _cache_set(cache_key, result)
                app.logger.info("LLM request served by %s", name)
                return result
        except Exception as exc:
            app.logger.warning("Provider %s failed: %s", name, exc)

    return "⚠️ All AI providers are currently busy. Please try again in a few minutes."


_MAX_FIELD_LEN = 4000  # max characters per input field


def _get_json(required_fields):
    data = request.get_json(silent=True) or {}
    missing = [field for field in required_fields if not str(data.get(field, "")).strip()]
    if missing:
        return None, jsonify(error=f"Missing required field(s): {', '.join(missing)}"), 400
    for field, val in data.items():
        if isinstance(val, str) and len(val) > _MAX_FIELD_LEN:
            return None, jsonify(error=f"Field '{field}' exceeds maximum length of {_MAX_FIELD_LEN} characters."), 400
    return data, None, None


def _internal_error():
    app.logger.exception("Unhandled route error")
    return jsonify(error="Internal server error. Please try again."), 500


# ── HTML ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>EduSafeAI Hub</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="AI-powered tools for K-12 educators worldwide.">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%8F%AB%3C/text%3E%3C/svg%3E">
<style>
:root{--gd:#1a472a;--gm:#2d6a4f;--gl:#52b788;--gp:#f0f7f0;--gb:#c8e6c9;--w:#fff;--gray:#666;--r:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:var(--gp);color:#222}
.header{background:linear-gradient(135deg,var(--gd),var(--gl));color:#fff;padding:32px 20px;text-align:center}
.header h1{font-size:2.2em;margin-bottom:8px;letter-spacing:1px}
.header p{font-size:1em;opacity:.92;max-width:580px;margin:0 auto}
.badges{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:14px}
.badge{background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:5px 14px;font-size:.82em}
.container{max-width:920px;margin:28px auto;padding:0 16px}
.tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(80px,1fr));gap:8px;margin-bottom:24px}
.tabs button{background:var(--gm);color:#fff;border:none;padding:10px 4px;border-radius:var(--r);cursor:pointer;font-size:11px;font-weight:600;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:3px;width:100%}
.tabs button:hover{background:var(--gd);transform:translateY(-2px)}
.tabs button.active{background:var(--gd);border-bottom:3px solid var(--gl)}
.tab-icon{font-size:1.4em}
.tab{display:none}.tab.active{display:block;animation:fadeIn .35s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--w);padding:28px;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.09);margin-bottom:4px}
.card h2{color:var(--gd);margin-bottom:8px;font-size:1.35em;display:flex;align-items:center;gap:8px}
.hint{color:var(--gray);font-size:.87em;margin-bottom:18px;background:#e8f5e9;padding:12px 14px;border-radius:0 10px 10px 0;border-left:4px solid var(--gl)}
.new-badge{background:#e8f5e9;color:#2d6a4f;border:1px solid #a5d6a7;border-radius:12px;padding:2px 8px;font-size:.7em;font-weight:bold;margin-left:8px;vertical-align:middle}
.form-row{display:grid;grid-template-columns:1fr;gap:14px;margin-bottom:8px}
@media(min-width:500px){.form-row.two{grid-template-columns:1fr 1fr}}
.field{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
label{font-weight:600;color:var(--gd);font-size:.9em;display:flex;align-items:center;gap:6px}
.tip{display:inline-block;background:var(--gm);color:#fff;border-radius:50%;width:17px;height:17px;font-size:.7em;text-align:center;line-height:17px;cursor:help;position:relative}
.tip:hover::after{content:attr(data-tip);position:absolute;left:22px;top:-4px;background:#333;color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;white-space:nowrap;z-index:10;font-weight:normal}
input,select,textarea{width:100%;padding:11px 13px;border:1.5px solid #ddd;border-radius:var(--r);font-size:14px;transition:border .2s;background:#fafafa}
input:focus,select:focus,textarea:focus{border-color:var(--gl);outline:none;background:#fff}
textarea{resize:vertical;min-height:100px}
.std-desc{background:#e8f5e9;border-left:3px solid var(--gl);padding:8px 12px;border-radius:0 8px 8px 0;font-size:13px;color:#2d4a35;margin-top:6px;display:none}
.std-desc.show{display:block}
.btn{background:linear-gradient(135deg,#43a047,#2d6a4f);color:#fff;border:none;padding:14px;width:100%;border-radius:var(--r);font-size:15px;cursor:pointer;margin:14px 0 8px;font-weight:bold;letter-spacing:.5px;transition:all .2s;box-shadow:0 3px 8px rgba(0,0,0,.15)}
.btn:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.2)}
.btn:disabled{opacity:.6;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:6px}
.output{background:#f6fdf6;border:1.5px solid var(--gb);border-radius:var(--r);padding:18px;min-height:90px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:8px;right:8px;background:var(--gm);color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
hr{border:none;border-top:1px solid #e0e0e0;margin:18px 0}
.footer{text-align:center;padding:28px 16px;color:var(--gray);font-size:13px;line-height:2;background:var(--w);border-radius:16px;margin-top:20px}
select optgroup{font-weight:bold;color:var(--gd)}
.char-counter{font-size:.78em;color:var(--gray);text-align:right;margin-top:2px}
</style>
</head>
<body>
<div class="header">
  <a href="/" style="color:inherit;text-decoration:none"><h1>🛡️ EduSafeAI Hub</h1></a>
  <p><b>AI tools for K-12 educators worldwide</b></p>
  <div class="badges">
    <span class="badge">🏫 Education-Ready</span>
    <span class="badge">♿ IEP & ELL Ready</span>
    <span class="badge">🔒 No Student Data Stored</span>
    <span class="badge">☁️ Multi-AI Powered</span>
  </div>
</div>

<div class="container">
  <div class="tabs" id="tool-tabs" role="tablist" aria-label="Tool categories">
    <button class="active" role="tab" aria-selected="true" aria-controls="lesson" data-tab="lesson"><span class="tab-icon">📖</span>Lesson</button>
    <button role="tab" aria-selected="false" aria-controls="feedback" data-tab="feedback"><span class="tab-icon">💬</span>Feedback</button>
    <button role="tab" aria-selected="false" aria-controls="diff" data-tab="diff"><span class="tab-icon">♿</span>IEP/ELL</button>
    <button role="tab" aria-selected="false" aria-controls="policy" data-tab="policy"><span class="tab-icon">📄</span>Policy</button>
    <button role="tab" aria-selected="false" aria-controls="email" data-tab="email"><span class="tab-icon">✉️</span>Email</button>
    <button role="tab" aria-selected="false" aria-controls="integrity" data-tab="integrity"><span class="tab-icon">🧪</span>AI Integrity</button>
    <button role="tab" aria-selected="false" aria-controls="assessment_prep" data-tab="assessment_prep"><span class="tab-icon">📊</span>Assessment Prep</button>
    <button role="tab" aria-selected="false" aria-controls="parent" data-tab="parent"><span class="tab-icon">🗣️</span>Parent Letter</button>
    <button role="tab" aria-selected="false" aria-controls="unit" data-tab="unit"><span class="tab-icon">📅</span>Unit Plan</button>
    <button role="tab" aria-selected="false" aria-controls="iep504" data-tab="iep504"><span class="tab-icon">🏫</span>504/IEP</button>
    <button role="tab" aria-selected="false" aria-controls="quiz" data-tab="quiz"><span class="tab-icon">❓</span>Quiz</button>
    <button role="tab" aria-selected="false" aria-controls="rubric" data-tab="rubric"><span class="tab-icon">🧾</span>Rubric</button>
    <button role="tab" aria-selected="false" aria-controls="refine" data-tab="refine"><span class="tab-icon">🔧</span>Improve</button>
    <button role="tab" aria-selected="false" aria-controls="sitefb" data-tab="sitefb"><span class="tab-icon">⭐</span>Contact Us</button>
  </div>

  <!-- 1. LESSON -->
  <div id="lesson" class="tab active"><div class="card">
    <h2>📖 AI Lesson Designer</h2>
    <p class="hint">Generate a complete standards-aligned lesson plan with hooks, activities, and exit tickets.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="l2" onchange="updateStandards()">
          <option value="K-2">K-2</option><option value="3-5">3-5</option>
          <option value="6-8" selected>6-8</option><option value="9-12">9-12</option>
        </select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="l3" onchange="updateStandards()">
          <option value="Social Studies" selected>Social Studies</option>
          <option value="ELA">ELA</option><option value="Science">Science</option>
          <option value="Math">Math</option><option value="Health">Health</option>
          <option value="World Languages">World Languages</option><option value="Tech/CS">Tech/CS</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label>Standard <span class="tip" data-tip="Select grade and subject first">?</span></label>
      <select id="l1" onchange="showStdDesc()">
        <option value="">-- Select grade & subject above first --</option>
      </select>
      <div id="std-desc" class="std-desc"></div>
    </div>
    <div class="field">
      <label>Duration</label>
      <select id="l4"><option selected>45 min</option><option>60 min</option><option>90 min</option><option>Block</option></select>
    </div>
    <button class="btn" id="lb" onclick="call('/lesson',{standard:g('l1'),grade:g('l2'),subject:g('l3'),duration:g('l4')},'lo','lb','🎯 Generate Full Lesson Plan')">🎯 Generate Full Lesson Plan</button>
    <div class="output-wrap"><div id="lo" class="output">Your lesson plan will appear here...</div><button class="copy-btn" onclick="cp('lo')">📋 Copy</button></div>
  </div></div>

  <!-- 2. FEEDBACK -->
  <div id="feedback" class="tab"><div class="card">
    <h2>💬 Student Feedback Generator</h2>
    <p class="hint">Get constructive, rubric-aligned feedback without rewriting student work.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="f1"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option></select>
      </div>
      <div class="field">
        <label>Rubric Type</label>
        <select id="f2"><option selected>General Writing</option><option>Social Studies DBQ</option><option>ELA Essay</option><option>Science Lab Report</option></select>
      </div>
    </div>
    <div class="field">
      <label>Student Work <span class="tip" data-tip="Paste student writing below">?</span></label>
      <textarea id="f3" rows="6" maxlength="4000" placeholder="Paste student work here..."></textarea>
      <span class="char-counter"></span>
    </div>
    <button class="btn" id="fb" onclick="call('/feedback',{work:g('f3'),grade:g('f1'),rubric:g('f2')},'fo','fb','💬 Generate Feedback')">💬 Generate Feedback</button>
    <div class="output-wrap"><div id="fo" class="output">Feedback will appear here...</div><button class="copy-btn" onclick="cp('fo')">📋 Copy</button></div>
  </div></div>

  <!-- 3. IEP/ELL -->
  <div id="diff" class="tab"><div class="card">
    <h2>♿ IEP/ELL Differentiator</h2>
    <p class="hint">Adapt any lesson for students with IEPs, ELL needs, ADHD, dyslexia, and more.</p>
    <hr>
    <div class="field">
      <label>Paste Your Lesson</label>
      <textarea id="d1" rows="5" maxlength="4000" placeholder="Paste any lesson plan here..."></textarea>
      <span class="char-counter"></span>
    </div>
    <div class="field">
      <label>Student Needs</label>
      <input id="d2" placeholder="e.g. ELL Level 2, ADHD, dyslexia, hearing impaired">
    </div>
    <button class="btn" id="db" onclick="call('/differentiate',{lesson:g('d1'),needs:g('d2')},'do','db','♿ Differentiate This Lesson')">♿ Differentiate This Lesson</button>
    <div class="output-wrap"><div id="do" class="output">Differentiated version will appear here...</div><button class="copy-btn" onclick="cp('do')">📋 Copy</button></div>
  </div></div>

  <!-- 4. POLICY -->
  <div id="policy" class="tab"><div class="card">
    <h2>📄 AI Policy Generator</h2>
    <p class="hint">Generate an official AI use policy for your school or district.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>School / District</label><input id="p1" placeholder="Enter your school name"></div>
      <div class="field"><label>School Year</label><input id="p2" value="2025-2026"></div>
    </div>
    <div class="form-row two">
      <div class="field">
        <label>Grade Band</label>
        <select id="p3"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option><option>All grades</option></select>
      </div>
      <div class="field"><label>Main Concerns</label><input id="p4" placeholder="e.g. plagiarism, data privacy, parent communication"></div>
    </div>
    <button class="btn" id="pb" onclick="call('/policy',{school:g('p1'),year:g('p2'),grade:g('p3'),concerns:g('p4')},'po','pb','📄 Generate Official Policy')">📄 Generate Official Policy</button>
    <div class="output-wrap"><div id="po" class="output">Policy will appear here...</div><button class="copy-btn" onclick="cp('po')">📋 Copy</button></div>
  </div></div>

  <!-- 5. EMAIL -->
  <div id="email" class="tab"><div class="card">
    <h2>✉️ Email Drafter</h2>
    <p class="hint">Draft professional emails to parents, admin, or colleagues in seconds.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Recipient</label>
        <select id="e1"><option selected>Parent</option><option>Principal</option><option>Colleague</option><option>Superintendent</option><option>IEP Team</option><option>School Board</option></select>
      </div>
      <div class="field">
        <label>Tone</label>
        <select id="e2"><option selected>professional</option><option>friendly</option><option>formal</option><option>urgent</option><option>empathetic</option></select>
      </div>
    </div>
    <div class="field"><label>What is the email about?</label><input id="e3" placeholder="e.g. Student used AI without permission on assignment"></div>
    <button class="btn" id="eb" onclick="call('/email',{recipient:g('e1'),tone:g('e2'),topic:g('e3')},'eo','eb','✉️ Draft Email')">✉️ Draft Email</button>
    <div class="output-wrap"><div id="eo" class="output">Email will appear here...</div><button class="copy-btn" onclick="cp('eo')">📋 Copy</button></div>
  </div></div>

  <!-- 6. AI INTEGRITY -->
  <div id="integrity" class="tab"><div class="card">
    <h2>🧪 AI Integrity Checker <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Paste student work to get an AI-risk analysis plus a conversation script to discuss it professionally.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="i1"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option></select>
      </div>
      <div class="field">
        <label>Assignment Type</label>
        <select id="i2"><option selected>Essay</option><option>Research Paper</option><option>Short Answer</option><option>Lab Report</option><option>Creative Writing</option></select>
      </div>
    </div>
    <div class="field"><label>Student Work</label><textarea id="i3" rows="7" maxlength="4000" placeholder="Paste student work here..."></textarea><span class="char-counter"></span></div>
    <button class="btn" id="ib" onclick="call('/integrity',{work:g('i3'),grade:g('i1'),type:g('i2')},'io','ib','🧪 Check AI Integrity')">🧪 Check AI Integrity</button>
    <div class="output-wrap"><div id="io" class="output">Analysis will appear here...</div><button class="copy-btn" onclick="cp('io')">📋 Copy</button></div>
  </div></div>

  <!-- 7. ASSESSMENT PREP -->
  <div id="assessment_prep" class="tab"><div class="card">
    <h2>📊 Standards-Based Assessment Prep <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Generate standards-aligned practice questions for any assessment framework.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="n1"><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option selected>8</option><option>9</option><option>10</option><option>11</option></select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="n2"><option selected>ELA</option><option>Math</option><option>Science (NGSS)</option></select>
      </div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Standard <span class="tip" data-tip="Leave blank for auto-select">?</span></label><input id="n3" placeholder="e.g. RL.8.1 or leave blank"></div>
      <div class="field">
        <label>Question Type</label>
        <select id="n4"><option selected>Multiple Choice</option><option>Short Answer</option><option>Evidence-Based</option><option>Mixed</option></select>
      </div>
    </div>
    <div class="field">
      <label>Number of Questions</label>
      <select id="n5"><option selected>5</option><option>10</option><option>15</option></select>
    </div>
    <button class="btn" id="nb" onclick="call('/assessment_prep',{grade:g('n1'),subject:g('n2'),standard:g('n3'),qtype:g('n4'),num:g('n5')},'no','nb','📊 Generate Assessment Practice')">📊 Generate Assessment Practice Questions</button>
    <div class="output-wrap"><div id="no" class="output">Practice questions will appear here...</div><button class="copy-btn" onclick="cp('no')">📋 Copy</button></div>
  </div></div>

  <!-- 8. PARENT LETTER -->
  <div id="parent" class="tab"><div class="card">
    <h2>🗣️ Parent Letter Generator <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Generate culturally appropriate parent letters in multiple languages.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Language</label>
        <select id="pl1"><option selected>English</option><option>Spanish</option><option>Portuguese</option><option>French</option><option>Chinese (Simplified)</option><option>Arabic</option><option>Haitian Creole</option></select>
      </div>
      <div class="field">
        <label>Letter Type</label>
        <select id="pl2"><option selected>Classroom AI Policy</option><option>Student Behavior</option><option>Academic Progress</option><option>IEP Meeting Invite</option><option>Field Trip Permission</option><option>Homework Policy</option></select>
      </div>
    </div>
    <div class="field"><label>Key Details</label><input id="pl3" placeholder="e.g. Meeting on March 5th at 3pm"></div>
    <div class="field"><label>Teacher Name & School</label><input id="pl4" placeholder="e.g. Ms. Johnson, Lincoln Elementary"></div>
    <button class="btn" id="plb" onclick="call('/parent_letter',{lang:g('pl1'),type:g('pl2'),details:g('pl3'),teacher:g('pl4')},'plo','plb','🗣️ Generate Parent Letter')">🗣️ Generate Parent Letter</button>
    <div class="output-wrap"><div id="plo" class="output">Parent letter will appear here...</div><button class="copy-btn" onclick="cp('plo')">📋 Copy</button></div>
  </div></div>

  <!-- 9. UNIT PLANNER -->
  <div id="unit" class="tab"><div class="card">
    <h2>📅 2-Week Unit Planner <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Plan a complete 2-week unit with daily breakdown, assessments, and differentiation.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Unit Topic</label><input id="u1" placeholder="e.g. American Revolution, Fractions, Ecosystems"></div>
      <div class="field"><label>Standard</label><input id="u2" placeholder="e.g. 6.1.8.HistoryCC.3 or local standard"></div>
    </div>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="u3"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="u4"><option selected>Social Studies</option><option>ELA</option><option>Science</option><option>Math</option><option>Health</option></select>
      </div>
    </div>
    <div class="field">
      <label>Class Duration</label>
      <select id="u5"><option selected>45 min</option><option>60 min</option><option>90 min</option><option>Block</option></select>
    </div>
    <button class="btn" id="ub" onclick="call('/unit_plan',{topic:g('u1'),standard:g('u2'),grade:g('u3'),subject:g('u4'),duration:g('u5')},'uo','ub','📅 Generate 2-Week Unit Plan')">📅 Generate 2-Week Unit Plan</button>
    <div class="output-wrap"><div id="uo" class="output">Unit plan will appear here...</div><button class="copy-btn" onclick="cp('uo')">📋 Copy</button></div>
  </div></div>

  <!-- 10. 504/IEP -->
  <div id="iep504" class="tab"><div class="card">
    <h2>🏫 504 vs IEP Helper <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Clarify 504 vs IEP differences and generate accommodations for any disability or need.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Plan Type</label>
        <select id="s1">
          <option selected>Explain difference: 504 vs IEP</option>
          <option>Generate 504 accommodations</option>
          <option>Generate IEP accommodations</option>
          <option>Generate both 504 & IEP options</option>
        </select>
      </div>
      <div class="field">
        <label>Grade</label>
        <select id="s2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select>
      </div>
    </div>
    <div class="field"><label>Student Disability / Need</label><input id="s3" placeholder="e.g. ADHD, anxiety, dyslexia, autism, hearing impaired"></div>
    <div class="field"><label>Subject Context</label><input id="s4" placeholder="e.g. ELA class, standardized testing, all subjects"></div>
    <button class="btn" id="sb" onclick="call('/iep504',{plan:g('s1'),grade:g('s2'),disability:g('s3'),context:g('s4')},'so','sb','🏫 Generate Accommodations')">🏫 Generate Accommodations</button>
    <div class="output-wrap"><div id="so" class="output">Accommodations will appear here...</div><button class="copy-btn" onclick="cp('so')">📋 Copy</button></div>
  </div></div>


  <!-- 11. QUIZ BUILDER -->
  <div id="quiz" class="tab"><div class="card">
    <h2>❓ Quiz Builder</h2>
    <p class="hint">Create standards-aligned formative quizzes with answer keys.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Topic</label><input id="q1" placeholder="e.g. Civil War causes"></div>
      <div class="field"><label>Grade</label><select id="q2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select></div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Question Type</label><select id="q3"><option selected>Mixed</option><option>Multiple Choice</option><option>Short Answer</option></select></div>
      <div class="field"><label>Number of Questions</label><select id="q4"><option>5</option><option selected>10</option><option>15</option></select></div>
    </div>
    <button class="btn" id="qb" onclick="call('/quiz',{topic:g('q1'),grade:g('q2'),qtype:g('q3'),num:g('q4')},'qo','qb','❓ Generate Quiz')">❓ Generate Quiz</button>
    <div class="output-wrap"><div id="qo" class="output">Quiz will appear here...</div><button class="copy-btn" onclick="cp('qo')">📋 Copy</button></div>
  </div></div>

  <!-- 12. RUBRIC BUILDER -->
  <div id="rubric" class="tab"><div class="card">
    <h2>🧾 Rubric Builder</h2>
    <p class="hint">Generate clear performance-level rubrics aligned to your assignment goals.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Assignment</label><input id="r1" placeholder="e.g. Argument essay"></div>
      <div class="field"><label>Grade</label><select id="r2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select></div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Criteria Count</label><select id="r3"><option>3</option><option selected>4</option><option>5</option></select></div>
      <div class="field"><label>Scale</label><select id="r4"><option selected>4-point</option><option>5-point</option></select></div>
    </div>
    <button class="btn" id="rb" onclick="call('/rubric',{assignment:g('r1'),grade:g('r2'),criteria:g('r3'),scale:g('r4')},'ro','rb','🧾 Generate Rubric')">🧾 Generate Rubric</button>
    <div class="output-wrap"><div id="ro" class="output">Rubric will appear here...</div><button class="copy-btn" onclick="cp('ro')">📋 Copy</button></div>
  </div></div>

  <!-- 13. IMPROVE AI RESPONSE -->
  <div id="refine" class="tab"><div class="card">
    <h2>🔧 Improve AI Response</h2>
    <p class="hint">Ask the AI to revise any answer — make it shorter, simpler, add accommodations, etc.</p>
    <hr>
    <div class="field"><label>AI Output to Improve</label><textarea id="rf1" rows="6" maxlength="4000" placeholder="Paste the AI response you want to improve..."></textarea><span class="char-counter"></span></div>
    <div class="field"><label>What should be fixed?</label><input id="rf2" placeholder="e.g. make it shorter, add accommodations, simpler language"></div>
    <button class="btn" id="rfb" onclick="call('/refine_response',{response:g('rf1'),request:g('rf2')},'rfo','rfb','🔧 Improve Response')">🔧 Improve Response</button>
    <div class="output-wrap"><div id="rfo" class="output">Refined response will appear here...</div><button class="copy-btn" onclick="cp('rfo')">📋 Copy</button></div>
  </div></div>

  <!-- 14. CONTACT US -->
  <div id="sitefb" class="tab"><div class="card">
    <h2>⭐ Contact Us</h2>
    <p>Have ideas, suggestions, or found a bug? We'd love to hear from you!</p>
    <hr>
    <p style="font-size:1.1em">📧 Email us at: <strong>admin@edusafeai.com</strong></p>
  </div></div>

</div>

<div class="footer">
  <strong>🛡️ EduSafeAI Hub</strong> | AI tools for K-12 educators worldwide <br>
  🔒 EduSafeAI does not store student data. Please do not enter personally identifiable student information.<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ AI-generated content may contain inaccuracies. 
    This tool is for informational purposes only and does not constitute legal, medical, or professional advice. 
    Always consult qualified professionals for official IEP, 504, or compliance decisions.
  </span>
</div>

<script>
const STANDARDS={
  "Social Studies":{"K-2":[{code:"6.1.2.CivicsPD.1",desc:"Civics: Students explain how classroom rules and school rules help everyone"},{code:"6.1.2.GeoPP.1",desc:"Geography: Describe how location affects daily life"},{code:"6.1.2.EconET.1",desc:"Economics: Explain how people earn, save, and spend money"},{code:"6.1.2.HistoryCC.1",desc:"History: Place events in chronological order using timelines"}],"3-5":[{code:"6.1.5.CivicsPD.1",desc:"Civics: Explain how democratic processes work in the US"},{code:"6.1.5.GeoPP.2",desc:"Geography: Describe how humans have modified the environment"},{code:"6.1.5.EconNE.1",desc:"Economics: Explain how supply and demand affect prices"},{code:"6.1.5.HistoryCC.3",desc:"History: Sequence key events of colonial America and American Revolution"}],"6-8":[{code:"6.1.8.CivicsPD.1",desc:"Civics: Analyze how democratic ideals are reflected in the Constitution"},{code:"6.1.8.CivicsPI.1",desc:"Civics: Describe how the three branches of government interact"},{code:"6.1.8.GeoPP.2",desc:"Geography: Explain how geography influenced westward expansion"},{code:"6.1.8.EconNE.1",desc:"Economics: Analyze causes and effects of the Industrial Revolution"},{code:"6.1.8.HistoryCC.3",desc:"History: Analyze causes and effects of the Civil War"}],"9-12":[{code:"6.1.12.CivicsPD.4",desc:"Civics: Analyze how Constitutional amendments expanded civil rights"},{code:"6.1.12.EconGE.1",desc:"Economics: Evaluate causes and effects of the Great Depression"},{code:"6.1.12.HistoryCC.7",desc:"History: Analyze US foreign policy during the Cold War"}]},
  "ELA":{"K-2":[{code:"RL.K.1",desc:"Reading Literature: With prompting, ask/answer questions about key details"},{code:"RL.1.1",desc:"Reading Literature: Ask and answer questions about key details"},{code:"W.1.3",desc:"Writing: Write narratives about two or more events with details"}],"3-5":[{code:"RL.3.3",desc:"Reading Literature: Describe characters and explain how actions contribute to plot"},{code:"RL.5.1",desc:"Reading Literature: Quote accurately when explaining what the text says"},{code:"W.4.1",desc:"Writing: Write opinion pieces supporting a point of view"}],"6-8":[{code:"RL.6.1",desc:"Reading Literature: Cite textual evidence to support analysis"},{code:"RL.8.1",desc:"Reading Literature: Cite evidence that most strongly supports an inference"},{code:"RI.8.1",desc:"Reading Informational Text: Cite textual evidence for analysis"},{code:"SL.8.1",desc:"Speaking/Listening: Engage effectively in collaborative discussions"},{code:"W.7.1",desc:"Writing: Write arguments to support claims with clear reasons"},{code:"W.8.2",desc:"Writing: Write explanatory texts to convey complex ideas"}],"9-12":[{code:"RL.9-10.1",desc:"Reading Literature: Cite strong textual evidence to support analysis"},{code:"W.9-10.1",desc:"Writing: Write arguments that introduce precise claims"},{code:"W.11-12.3",desc:"Writing: Write narratives using pacing, description, reflection"}]},
  "Science":{"K-2":[{code:"K-PS2-1",desc:"NGSS Forces & Motion: Compare effects of different forces"},{code:"K-ESS2-1",desc:"NGSS Earth Science: Describe weather patterns over time"}],"3-5":[{code:"3-LS1-1",desc:"NGSS Life Science: Describe that organisms have unique life cycles"},{code:"4-PS3-2",desc:"NGSS Energy: Explain that energy can be transferred in various ways"}],"6-8":[{code:"MS-PS1-1",desc:"NGSS Matter: Describe atomic composition of molecules"},{code:"MS-LS1-1",desc:"NGSS Life Science: Provide evidence that living things are made of cells"},{code:"MS-ESS2-1",desc:"NGSS Earth Science: Describe cycling of Earth's materials"}],"9-12":[{code:"HS-PS1-1",desc:"NGSS Chemistry: Use periodic table to predict properties of elements"},{code:"HS-LS1-1",desc:"NGSS Biology: Explain how DNA determines proteins"},{code:"HS-ESS2-2",desc:"NGSS Earth Science: Forecast natural hazards from data"}]},
  "Math":{"K-2":[{code:"K.CC.A.1",desc:"Counting: Count to 100 by ones and tens"},{code:"1.OA.A.1",desc:"Operations: Use addition and subtraction within 20"}],"3-5":[{code:"3.OA.A.1",desc:"Operations: Interpret products of whole numbers"},{code:"4.NF.A.1",desc:"Fractions: Explain equivalent fractions using models"},{code:"5.NBT.A.1",desc:"Number: Digit in one place is 10 times digit to its right"}],"6-8":[{code:"6.RP.A.1",desc:"Ratios: Understand concept of ratio and use ratio language"},{code:"7.NS.A.1",desc:"Number System: Add and subtract rational numbers"},{code:"8.EE.B.5",desc:"Expressions: Graph proportional relationships, interpreting slope"},{code:"8.G.A.1",desc:"Geometry: Verify properties of rotations, reflections, translations"},{code:"8.F.A.1",desc:"Functions: Understand input-output relationships"}],"9-12":[{code:"HSA.CED.A.1",desc:"Algebra: Create equations and inequalities to solve problems"},{code:"HSF.IF.A.1",desc:"Functions: Understand that a function assigns exactly one output"},{code:"HSS.ID.A.1",desc:"Statistics: Represent data with dot plots, histograms, box plots"}]},
  "Health":{"K-2":[{code:"2.1.2.A.1",desc:"Personal Health: Identify behaviors that promote health and safety"}],"3-5":[{code:"2.1.5.B.1",desc:"Physical Activity: Explain how physical activity improves health"}],"6-8":[{code:"2.1.8.C.1",desc:"Social/Emotional: Analyze how stress affects physical and mental health"}],"9-12":[{code:"2.1.12.D.1",desc:"Health Decisions: Evaluate how personal choices affect long-term health"}]},
  "World Languages":{"K-2":[{code:"7.1.NM.IPRET.1",desc:"Interpretive: Recognize familiar words and greetings in the target language"}],"3-5":[{code:"7.1.NM.IPERS.2",desc:"Interpersonal: Exchange basic personal information using memorized phrases"}],"6-8":[{code:"7.1.NM.PRSNT.3",desc:"Presentational: Present simple information using practiced language"},{code:"7.1.NM.IPRET.4",desc:"Interpretive: Identify main idea in short, familiar texts/audio"}],"9-12":[{code:"7.1.NH.IPERS.5",desc:"Interpersonal: Sustain short conversations on familiar topics"},{code:"7.1.NH.PRSNT.2",desc:"Presentational: Create organized oral/written messages for specific audience"}]},
    "Tech/CS":{"K-2":[{code:"8.1.2.CS.1",desc:"CS: Select and use hardware and software to complete basic tasks"}],"3-5":[{code:"8.1.5.AP.1",desc:"CS: Create programs with sequences, events, and loops"}],"6-8":[{code:"8.1.8.AP.2",desc:"CS: Create clearly named variables to store and use data"}],"9-12":[{code:"8.1.12.DA.1",desc:"Data: Create data visualizations to communicate insights"}]}
};

function g(id){return document.getElementById(id).value;}

function updateStandards(){
  const grade=g('l2'),subject=g('l3'),sel=document.getElementById('l1'),desc=document.getElementById('std-desc');
  sel.innerHTML='';desc.className='std-desc';desc.textContent='';
  const list=(STANDARDS[subject]||{})[grade]||[];
  if(!list.length){sel.innerHTML='<option value="">-- No standards available --</option>';return;}
  list.forEach((s,i)=>{const o=document.createElement('option');o.value=s.code;o.textContent=s.code;if(i===0)o.selected=true;sel.appendChild(o);});
  showStdDesc();
}

function showStdDesc(){
  const grade=g('l2'),subject=g('l3'),code=g('l1'),desc=document.getElementById('std-desc');
  const list=(STANDARDS[subject]||{})[grade]||[];
  const found=list.find(s=>s.code===code);
  if(found){desc.textContent=found.desc;desc.className='std-desc show';}
  else{desc.className='std-desc';}
}

window.onload=()=>{updateStandards();};

function show(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button[role="tab"]').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');});
  const target=document.getElementById(tab);
  if(target){target.classList.add('active');}
  if(btn){btn.classList.add('active');btn.setAttribute('aria-selected','true');}
}

function _fallbackCopy(text){
  const ta=document.createElement('textarea');
  ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.focus();ta.select();
  try{document.execCommand('copy');}catch(e){}
  document.body.removeChild(ta);
}

function cp(id){
  const text=document.getElementById(id).innerText;
  const btn=document.querySelector('#'+id).parentNode.querySelector('.copy-btn');
  const done=()=>{btn.textContent='✅ Copied!';setTimeout(()=>btn.textContent='📋 Copy',2000);};
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(()=>{_fallbackCopy(text);done();});
  }else{_fallbackCopy(text);done();}
}

let focusMode='general';

document.addEventListener('DOMContentLoaded',()=>{
  // Tab delegation
  const tabList=document.getElementById('tool-tabs');
  tabList.addEventListener('click',e=>{
    const btn=e.target.closest('button[data-tab]');
    if(btn)show(btn.dataset.tab,btn);
  });
  // Keyboard navigation for tabs (ARIA tablist pattern)
  tabList.addEventListener('keydown',e=>{
    const tabs=Array.from(tabList.querySelectorAll('button[role="tab"]'));
    const idx=tabs.indexOf(document.activeElement);
    if(idx===-1)return;
    let next=-1;
    if(e.key==='ArrowRight'||e.key==='ArrowDown')next=(idx+1)%tabs.length;
    else if(e.key==='ArrowLeft'||e.key==='ArrowUp')next=(idx-1+tabs.length)%tabs.length;
    else if(e.key==='Home')next=0;
    else if(e.key==='End')next=tabs.length-1;
    if(next!==-1){e.preventDefault();tabs[next].focus();show(tabs[next].dataset.tab,tabs[next]);}
  });
  // Character counters
  document.querySelectorAll('textarea[maxlength]').forEach(ta=>{
    const counter=ta.nextElementSibling;
    if(counter&&counter.classList.contains('char-counter')){
      const max=ta.getAttribute('maxlength');
      counter.textContent=`0 / ${max}`;
      ta.addEventListener('input',()=>{counter.textContent=`${ta.value.length} / ${max}`;});
    }
  });
});

async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId),btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  out.textContent='⏳ AI is thinking...';
  try{
    const payload={...data,focus:focusMode};
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(!r.ok){
      let msg='Request failed.';
      try{const ej=await r.json();msg=ej.error||ej.result||msg;}catch(_){const t=await r.text();msg=t.substring(0,300)||msg;}
      out.textContent='❌ '+msg;
      return;
    }
    const j=await r.json();
    out.textContent=j.result || j.error || 'No response content.';
  }catch(e){
    out.textContent='❌ Error: '+e.message;
  }finally{
    btn.disabled=false;btn.textContent=label;
  }
}
</script>
</body>
</html>"""

# ── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/health')
def health():
    return jsonify(status='ok')

@app.route('/lesson', methods=['POST'])
def do_lesson():
    try:
        d, err, code = _get_json(["standard", "grade", "subject", "duration"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Expert {d['grade']} {d['subject']} teacher.",
            f"Create a {d['duration']} lesson for standard/topic: {d['standard']}.\n\n🎣 HOOK\n📚 MAIN ACTIVITY (ELL/IEP notes)\n🔍 ORIGINALITY CHECK\n🎯 EXIT TICKET",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/feedback', methods=['POST'])
def do_feedback():
    try:
        d, err, code = _get_json(["work", "grade", "rubric"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Encouraging {d['grade']} teacher. NEVER rewrite student work.",
            f"Rubric: {d['rubric']}\n\nStudent work:\n{d['work']}\n\n💪 STRENGTH\n📖 EVIDENCE\n🧠 REASONING\n🎯 NEXT STEP\n⚠️ ACADEMIC INTEGRITY NOTE",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/differentiate', methods=['POST'])
def do_diff():
    try:
        d, err, code = _get_json(["lesson", "needs"])
        if err:
            return err, code
        return jsonify(result=llm(
            "SPED and ELL expert teacher.",
            f"Adapt this lesson for: {d['needs']}\n\n{d['lesson']}\n\n📝 SIMPLIFIED VERSION\n🖼️ VISUAL AIDS\n🪜 SCAFFOLDS\n📊 MODIFIED ASSESSMENT\n⏱️ EXTENDED TIME NOTES",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/policy', methods=['POST'])
def do_policy():
    try:
        d, err, code = _get_json(["school", "year", "grade", "concerns"])
        if err:
            return err, code
        return jsonify(result=llm(
            "School administrator and education policy expert.",
            f"Write official AI use policy for {d['school']} | Year: {d['year']} | Grade: {d['grade']} | Concerns: {d['concerns']}\n\n📌 PURPOSE\n📋 SCOPE\n👨‍🎓 STUDENT GUIDELINES\n👩‍🏫 TEACHER RESPONSIBILITIES\n👨‍👩‍👧 PARENT COMMUNICATION\n⚠️ CONSEQUENCES\n📅 REVIEW DATE",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/email', methods=['POST'])
def do_email():
    try:
        d, err, code = _get_json(["recipient", "tone", "topic"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Professional educator communication expert.",
            f"Write a {d['tone']} email to {d['recipient']} about: {d['topic']}.\nInclude: Subject line, greeting, 2-3 paragraphs, professional closing. Under 200 words.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/integrity', methods=['POST'])
def do_integrity():
    try:
        d, err, code = _get_json(["work", "grade", "type"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Educator and academic integrity specialist.",
            f"Grade: {d['grade']} | Assignment: {d['type']}\n\nStudent work:\n{d['work']}\n\n🔍 AI-RISK ASSESSMENT (High/Medium/Low) with reasons\n📝 SUSPICIOUS PHRASES\n✅ LIKELY ORIGINAL ELEMENTS\n💬 CONVERSATION SCRIPT for teacher\n📋 NEXT STEPS\n⚠️ NOTE: Advisory only — not proof of AI use.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/assessment_prep', methods=['POST'])
def do_assessment_prep():
    try:
        d, err, code = _get_json(["grade", "subject", "qtype", "num"])
        if err:
            return err, code
        standard = d.get('standard', '').strip()
        std = standard if standard else "appropriate grade-level standard"
        return jsonify(result=llm(
            "Standards-based assessment specialist.",
            f"Create {d['num']} {d['qtype']} practice questions. Grade: {d['grade']} | Subject: {d['subject']} | Standard: {std}\n\nFor each:\n❓ QUESTION\n🅐 Answer choices (if MC)\n✅ CORRECT ANSWER\n💡 EXPLANATION\n📋 STANDARD ALIGNMENT",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/parent_letter', methods=['POST'])
def do_parent():
    try:
        d, err, code = _get_json(["type", "lang", "details", "teacher"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Professional educator writing parent communications in {d['lang']}. Culturally responsive tone.",
            f"Write a {d['type']} parent letter in {d['lang']}.\nDetails: {d['details']}\nFrom: {d['teacher']}\nInclude: Date, greeting, clear explanation, action needed, contact info, closing. Under 250 words. Write ONLY in {d['lang']}.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/unit_plan', methods=['POST'])
def do_unit():
    try:
        d, err, code = _get_json(["topic", "standard", "grade", "subject", "duration"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Expert {d['grade']} {d['subject']} curriculum designer.",
            f"Create a 2-week unit plan for: {d['topic']}\nStandard: {d['standard']} | Grade: {d['grade']} | Subject: {d['subject']} | Duration: {d['duration']}\n\n📌 UNIT OVERVIEW\n📅 WEEK 1 (Day 1–5)\n📅 WEEK 2 (Day 6–10)\n📊 ASSESSMENTS\n♿ DIFFERENTIATION\n📚 RESOURCES",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/iep504', methods=['POST'])
def do_iep504():
    try:
        d, err, code = _get_json(["plan", "grade", "disability", "context"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Special education expert. Knowledge of IDEA and Section 504.",
            f"Request: {d['plan']} | Grade: {d['grade']} | Need: {d['disability']} | Context: {d['context']}\n\n📋 PLAIN LANGUAGE EXPLANATION\n⚖️ LEGAL BASIS\n✅ SPECIFIC ACCOMMODATIONS (at least 8)\n🎯 CLASSROOM STRATEGIES\n👨‍👩‍👧 PARENT COMMUNICATION TIPS\n⚠️ Always consult your Child Study Team for official plans.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/quiz', methods=['POST'])
def do_quiz():
    try:
        d, err, code = _get_json(["topic", "grade", "qtype", "num"])
        if err:
            return err, code
        return jsonify(result=llm(
            "K-12 assessment specialist.",
            f"Create a {d['qtype']} quiz with {d['num']} questions for grade {d['grade']} on topic: {d['topic']}. Include answer key and short rationale.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/rubric', methods=['POST'])
def do_rubric():
    try:
        d, err, code = _get_json(["assignment", "grade", "criteria", "scale"])
        if err:
            return err, code
        return jsonify(result=llm(
            "K-12 instructional coach and rubric designer.",
            f"Build a {d['scale']} rubric for grade {d['grade']} assignment: {d['assignment']}. Include {d['criteria']} criteria with clear performance descriptors.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


@app.route('/refine_response', methods=['POST'])
def do_refine_response():
    try:
        d, err, code = _get_json(["response", "request"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Instructional writing coach for teachers.",
            f"Revise the following AI response based on teacher request.\n\nORIGINAL RESPONSE:\n{d['response']}\n\nTEACHER REQUEST:\n{d['request']}\n\nReturn improved version plus a brief list of what changed.",
            d.get('focus', 'general'),
        ))
    except Exception:
        return _internal_error()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
