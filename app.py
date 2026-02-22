"""
Secret Helper — Producer.ai-style Chat UI  v3.0
===================================================
Run:  python app.py
Open: http://localhost:7860

Conversational interface — type what you want, watch it build in real time.
"""
import os
import time
import threading

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import gradio as gr

import config
from pipeline import lyrics_gen, mixer, music_gen, prompt_parser, vocal_gen, secret_helper
from pipeline import history as hist
from pipeline.vocal_gen import VOICE_PRESETS

os.makedirs("output", exist_ok=True)


def _check_ollama() -> tuple:
    """Returns (online, model_ready, message)."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        found = any(config.OLLAMA_MODEL.split(":")[0] in m for m in models)
        if found:
            return True, True, f"Ollama OK — {config.OLLAMA_MODEL} ready"
        return True, False, (
            f"Ollama running but model not found — "
            f"run: ollama pull {config.OLLAMA_MODEL}"
        )
    except Exception:
        return False, False, "Ollama offline — run: ollama serve"


_OL_ONLINE, _OL_MODEL, _OL_MSG = _check_ollama()
print(f"[ollama] {_OL_MSG}")

if _OL_ONLINE and _OL_MODEL:
    _OLLAMA_BANNER = (
        f'<div id="ollama-banner" style="text-align:center;font-size:0.68rem;'
        f'color:#00e5ff;opacity:0.5;padding:0.2rem 0;">● {config.OLLAMA_MODEL} ready</div>'
    )
else:
    _color = "#ffaa00" if _OL_ONLINE else "#ff5555"
    _OLLAMA_BANNER = (
        f'<div id="ollama-banner" style="text-align:center;font-size:0.72rem;'
        f'font-weight:700;color:{_color};padding:0.3rem 0.5rem;'
        f'background:#1a0a00;border-radius:6px;margin:0 1rem 0.3rem;">⚠ {_OL_MSG}</div>'
    )

VOICE_OPTIONS = list(VOICE_PRESETS.keys())
GENRE_OPTIONS = [
    "auto",
    # Hip-Hop / Urban
    "hip-hop", "boom-bap", "trap", "drill", "lo-fi",
    # Latin
    "reggaeton", "salsa", "bachata", "merengue", "cumbia", "latin pop",
    # Pop / Rock
    "pop", "rock", "indie", "punk", "metal", "alternative",
    # Electronic
    "electronic", "house", "techno", "dubstep", "drum & bass", "synthwave", "ambient",
    # Soul / R&B
    "r&b", "soul", "funk", "blues", "gospel",
    # World / Other
    "jazz", "classical", "reggae", "dancehall", "afrobeats",
    "bossa nova", "folk", "country", "k-pop", "disco",
]
MODEL_OPTIONS = ["small", "medium", "large"]

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #020d10 !important;
    color: #f0f0ff !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container { max-width: 700px !important; margin: 0 auto !important; }
footer { display: none !important; }
.built-with { display: none !important; }

/* ── Hero ── */
#hero {
    text-align: center;
    padding: 5rem 1rem 2.5rem;
}
#hero-icon {
    font-size: 2.2rem;
    margin-bottom: 0.7rem;
    background: linear-gradient(135deg, #00e5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero h2 {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #00e5ff, #7c3aed, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero p  { color: #4a5070; font-size: 0.88rem; }

/* ── Chat display ── */
#chat-display {
    padding: 0 1.2rem 1rem;
    overflow-y: auto;
    height: 60vh;        /* JS will override this immediately */
    min-height: 200px;
    scroll-behavior: smooth;
}

/* User bubble */
.user-wrap { display: flex; justify-content: flex-end; margin: 0.9rem 0; }
.user-bubble {
    background: linear-gradient(135deg, #0d1a2e, #1a0d2e);
    border: 1px solid #1e3060;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 1.1rem;
    max-width: 78%;
    font-size: 0.95rem;
    line-height: 1.45;
    color: #e8e8f0;
}

/* Bot area */
.bot-msg { margin: 0.5rem 0; }

/* Status line */
.mas-status {
    font-family: 'Consolas', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #9b5de5;
    margin: 0.7rem 0 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    opacity: 0.85;
}
.mas-dot {
    width: 18px; height: 18px;
    background: linear-gradient(135deg, #00e5ff22, #9b5de522);
    border: 1px solid #9b5de544;
    border-radius: 4px;
    font-size: 0.5rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #9b5de5;
    flex-shrink: 0;
}

/* Info cards */
.mas-card {
    background: #001a1f;
    border: 1px solid #003a45;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 0.3rem 0;
}
.mas-card-header {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #9b5de5;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 0.4rem;
}
.mas-card-body {
    color: #cceeff;
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* Progress card */
.mas-progress {
    background: #001a1f;
    border: 1px solid #003a45;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.mas-pct   { font-size: 1rem; font-weight: 700; min-width: 2.8rem; color: #9b5de5; }
.mas-ptitle { font-size: 0.88rem; font-weight: 600; color: #cceeff; }
.mas-psub  { font-size: 0.72rem; color: #2a6070; margin-top: 0.1rem; }

/* Narration text */
.mas-narration {
    font-size: 0.95rem;
    color: #cceeff;
    line-height: 1.6;
    margin: 0.55rem 0;
}

/* Suggestion chips */
.mas-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.55rem; }
.mas-chip {
    background: #001a1f;
    border: 1px solid #003a45;
    border-radius: 6px;
    padding: 0.32rem 0.65rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #2a8090;
    text-transform: uppercase;
    cursor: pointer;
    user-select: none;
    transition: border-color 0.15s, color 0.15s;
}
.mas-chip:hover { border-color: #00e5ff; color: #9b5de5; }

/* ── Bottom area (fixed to viewport bottom) ── */
#bottom-area {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(700px, 100vw) !important;
    background: #0a0a0f !important;
    z-index: 1000 !important;
    border-top: 1px solid #1e2a4a !important;
    padding: 0.2rem 0 0.6rem !important;
}
/* chat-display height is set dynamically by JS based on bottom-area size */

/* ── Scroll buttons ── */
.sh-scroll-btn {
    position: fixed;
    right: 18px;
    background: #0f0f1e;
    border: 1px solid #1e2a4a;
    border-radius: 8px;
    width: 34px;
    height: 34px;
    color: #00e5ff;
    font-size: 1.1rem;
    line-height: 1;
    cursor: pointer;
    z-index: 1001;
    opacity: 0.75;
    transition: opacity 0.15s, border-color 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
.sh-scroll-btn:hover { opacity: 1; border-color: #00e5ff; }
#disclaimer-text {
    text-align: center;
    font-size: 0.7rem;
    color: #00e5ff;
    opacity: 0.45;
    padding: 0.35rem 0;
}

/* Settings row */
label, .label-wrap span {
    color: #00e5ff !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.wrap-inner, select {
    background: #0f0f1e !important;
    border: 1px solid #1e2a4a !important;
    color: #c8d0f0 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}
input[type="radio"], input[type="checkbox"] { accent-color: #7c3aed !important; }

/* Input bar */
#input-bar { display: flex; gap: 0.5rem; align-items: flex-end; padding: 0.4rem 1rem 0.7rem; }
#prompt-box textarea {
    background: #0f0f1e !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 12px !important;
    color: #f0f0ff !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    resize: none !important;
}
#prompt-box textarea:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.1) !important;
    outline: none !important;
}
#prompt-box textarea::placeholder { color: rgba(0,229,255,0.4) !important; }

#send-btn {
    background: linear-gradient(135deg, #00bcd4, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    min-width: 46px !important;
    height: 46px !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
    transition: opacity 0.15s !important;
}
#send-btn:hover { opacity: 0.82 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1e2a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed44; }

/* Audio output */
audio { width: 100% !important; accent-color: #00e5ff !important; }
.audio-container, .waveform-container {
    background: #0f0f1e !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 10px !important;
    margin: 0 1.2rem 0.5rem !important;
}

/* ── Custom Lyrics accordion ── */
#lyrics-accordion {
    background: #0a0a0f !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 10px !important;
    margin: 0 1.2rem 0.5rem !important;
}
#lyrics-accordion .label-wrap span { color: #9b5de5 !important; font-size: 0.75rem !important; }
#custom-lyrics-box textarea {
    background: #0f0f1e !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 8px !important;
    color: #f0f0ff !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    font-family: 'Consolas', monospace !important;
}
#custom-lyrics-box textarea::placeholder { color: rgba(0,229,255,0.3) !important; }

/* ── Secret Helper panel ── */
#helper-actions, #quick-actions {
    padding: 0 1.2rem;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.helper-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #f0f0ff;
    margin: 0.6rem 0 0.1rem;
    letter-spacing: 0.02em;
}
.helper-meta {
    font-size: 0.72rem;
    color: #00e5ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    opacity: 0.75;
}
.helper-question {
    background: #1a0d2e;
    border: 1px solid #7c3aed44;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    color: #c8a8ff;
    font-size: 0.88rem;
    margin: 0.4rem 0;
}
#apply-btn {
    background: linear-gradient(135deg, #00bcd4, #7c3aed) !important;
    border: none !important; border-radius: 8px !important;
    color: #fff !important; font-weight: 700 !important;
    height: 38px !important; cursor: pointer !important;
}
#revise-btn, .regen-btn {
    background: #0f0f1e !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 8px !important; color: #c8d0f0 !important;
    font-weight: 600 !important; height: 38px !important;
    cursor: pointer !important; font-size: 0.82rem !important;
}
#revise-btn:hover, .regen-btn:hover {
    border-color: #00e5ff !important; color: #00e5ff !important;
}
#revise-box textarea {
    background: #0f0f1e !important; border: 1px solid #1e2a4a !important;
    border-radius: 8px !important; color: #f0f0ff !important;
    font-size: 0.88rem !important; padding: 0.5rem 0.75rem !important;
    resize: none !important;
}
#helper-btn {
    background: linear-gradient(135deg, #7c3aed, #00bcd4) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-size: 1rem !important;
    font-weight: 800 !important; min-width: 46px !important;
    height: 46px !important; cursor: pointer !important;
    flex-shrink: 0 !important; transition: opacity 0.15s !important;
}
#helper-btn:hover { opacity: 0.82 !important; }

/* ── Floating tooltip ── */
#sh-tip {
    position: fixed;
    background: #0a0a1a;
    border: 1px solid #00e5ff66;
    color: #00e5ff;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.28rem 0.6rem;
    border-radius: 6px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.1s;
    z-index: 99999;
    transform: translateX(-50%);
}
"""

# ── Auto-scroll JS ────────────────────────────────────────────────────────────

JS_SCROLL = """
(function() {
    const TIPS = {
        'send-btn':     'Generate song',
        'helper-btn':   'Ask Secret Helper — AI lyrics, structure & production notes',
        'apply-btn':    'Apply song settings + lyrics to the generator',
        'revise-btn':   'Send revision to Secret Helper',
        'regen-hook':   'Regenerate the Hook / Chorus',
        'regen-v1':     'Regenerate Verse 1',
        'regen-v2':     'Regenerate Verse 2',
        'regen-bridge': 'Regenerate the Bridge',
        'regen-sound':  'Regenerate the sound description',
    };

    // Tooltip
    let tip = document.getElementById('sh-tip');
    if (!tip) {
        tip = document.createElement('div');
        tip.id = 'sh-tip';
        document.body.appendChild(tip);
    }
    const show = (e) => {
        const text = e.currentTarget._shTip;
        if (!text) return;
        const r = e.currentTarget.getBoundingClientRect();
        tip.textContent = text;
        tip.style.left = (r.left + r.width / 2) + 'px';
        tip.style.top  = (r.top - 38) + 'px';
        tip.style.opacity = '1';
    };
    const hide = () => { tip.style.opacity = '0'; };
    const applyTips = () => {
        Object.entries(TIPS).forEach(([id, text]) => {
            const wrapper = document.getElementById(id);
            if (!wrapper) return;
            const btn = wrapper.querySelector('button') || wrapper;
            if (btn._shTip) return;
            btn._shTip = text;
            btn.addEventListener('mouseenter', show);
            btn.addEventListener('mouseleave', hide);
        });
    };

    // Dynamically size chat-display to fit exactly above the bottom bar
    const syncHeight = (chatEl, bottomEl) => {
        const bh = bottomEl.offsetHeight;
        chatEl.style.height = (window.innerHeight - bh) + 'px';
        chatEl.style.paddingBottom = '1rem';
    };

    // Scroll buttons
    const addScrollBtns = (chatEl, bottomEl) => {
        if (document.getElementById('sh-scroll-down')) return;

        const makeBtn = (id, label) => {
            const b = document.createElement('button');
            b.id = id;
            b.textContent = label;
            b.className = 'sh-scroll-btn';
            document.body.appendChild(b);
            return b;
        };

        const downBtn = makeBtn('sh-scroll-down', '↓');
        const upBtn   = makeBtn('sh-scroll-up',   '↑');

        downBtn.title = 'Scroll to bottom';
        upBtn.title   = 'Scroll to top';

        downBtn.onclick = () => { chatEl.scrollTop = chatEl.scrollHeight; };
        upBtn.onclick   = () => { chatEl.scrollTop = 0; };

        const positionBtns = () => {
            const bh = bottomEl.offsetHeight;
            downBtn.style.bottom = (bh + 12) + 'px';
            upBtn.style.bottom   = (bh + 54) + 'px';
            syncHeight(chatEl, bottomEl);
        };

        const ro = new ResizeObserver(positionBtns);
        ro.observe(bottomEl);
        window.addEventListener('resize', positionBtns);
        positionBtns();
    };

    const init = () => {
        const chatEl   = document.getElementById('chat-display');
        const bottomEl = document.getElementById('bottom-area');
        if (!chatEl || !bottomEl) { setTimeout(init, 500); return; }

        // Auto-scroll on new messages
        const obs = new MutationObserver(() => { chatEl.scrollTop = chatEl.scrollHeight; });
        obs.observe(chatEl, { childList: true, subtree: true });

        addScrollBtns(chatEl, bottomEl);
        applyTips();
        setTimeout(applyTips, 2000);
        setTimeout(applyTips, 5000);
    };
    setTimeout(init, 500);
})();
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────

HERO_HTML = """
<div id="hero">
  <div id="hero-icon">■ ■ ■</div>
  <h2>Secret Helper</h2>
  <p>Your friend in the studio</p>
</div>
"""

MOOD_NARRATIONS = {
    "chill":        "I'm laying down something smooth and mellow — minimal beats, soft textures.",
    "happy":        "I'm building something bright and upbeat — energy that makes you move.",
    "sad":          "I'm crafting something heavy and melancholic — dusty chops, slow rhythm.",
    "energetic":    "I'm pushing the tempo hard — high energy, punchy drums, wall of sound.",
    "romantic":     "I'm going warm and tender — lush chords, soft groove.",
    "dark":         "I'm going deep and mysterious — minor keys, haunting atmosphere.",
    "motivational": "I'm building something powerful — rising energy, anthemic feel.",
}

SUGGESTIONS = {
    "lo-fi":     ["ADD VINYL CRACKLE", "SLOWER BPM", "ADD RAIN SOUNDS", "MAKE IT SADDER"],
    "pop":       ["ADD A HOOK", "FASTER TEMPO", "MAKE IT ROMANTIC", "TRY FEMALE VOCALS"],
    "rock":      ["ADD GUITAR SOLO", "HEAVIER DRUMS", "MAKE IT EPIC", "ADD DISTORTION"],
    "jazz":      ["ADD JAZZ TRUMPET", "SLOWER SWING", "MORE BLUESY", "ADD PIANO SOLO"],
    "hip-hop":   ["ADD BOOM BAP", "CHANGE THE TEMPO", "MORE BASS", "ADD TRAP HATS"],
    "electronic":["ADD A DROP", "MORE REVERB", "MAKE IT AMBIENT", "ADD SYNTH LEAD"],
    "default":   ["CHANGE THE TEMPO", "DIFFERENT VOICE", "MAKE IT DARKER", "ADD INSTRUMENTS"],
}


def _status(text):
    return f'<div class="mas-status"><span class="mas-dot">&#9632;</span>{text}</div>'


def _card(header, body):
    safe = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<div class="mas-card">'
            f'<div class="mas-card-header">{header}</div>'
            f'<div class="mas-card-body">{safe}</div>'
            f'</div>')


def _prog(pct, title, sub):
    return (f'<div class="mas-progress">'
            f'<div class="mas-pct">{pct}%</div>'
            f'<div><div class="mas-ptitle">{title}</div>'
            f'<div class="mas-psub">{sub}</div></div>'
            f'</div>')


def _chips(items):
    inner = "".join(f'<span class="mas-chip">{c}</span>' for c in items)
    return f'<div class="mas-chips">{inner}</div>'


def _build(messages):
    if not messages:
        return HERO_HTML
    parts = []
    for role, content in messages:
        if role == "user":
            parts.append(
                f'<div class="user-wrap"><div class="user-bubble">{content}</div></div>'
            )
        else:
            parts.append(f'<div class="bot-msg">{content}</div>')
    return '<div id="chat-messages">' + "".join(parts) + "</div>"


# ── Secret Helper HTML builder ────────────────────────────────────────────────

def _build_helper_card(result: dict) -> str:
    song  = result.get("song", {})
    lyr   = result.get("lyrics", {})
    prod  = result.get("production_notes", {})
    msg   = result.get("assistant_message", "")
    q     = result.get("clarifying_question", "")

    parts = []
    if msg:
        parts.append(f'<div class="mas-narration">{msg}</div>')
    if result.get("need_clarification") and q:
        parts.append(f'<div class="helper-question">❓ {q}</div>')
        return "".join(parts)

    title = song.get("title", "")
    genre = song.get("genre", "")
    bpm   = song.get("bpm", "")
    voice = song.get("voice", "")
    tags  = ", ".join(song.get("mood_tags", []))

    if title:
        parts.append(f'<div class="helper-title">{title}</div>')
    meta = " · ".join(filter(None, [genre, f"{bpm} bpm" if bpm else "", voice, tags]))
    if meta:
        parts.append(f'<div class="helper-meta">{meta}</div>')

    sound = song.get("sound_description", "")
    if sound:
        parts.append(_card("🔊 SOUND", sound))

    text = lyr.get("text", "")
    if text:
        parts.append(_card("📝 LYRICS", text))

    arr = prod.get("arrangement", "")
    mix = prod.get("mix_notes", "")
    if arr or mix:
        parts.append(_card("🎚 PRODUCTION", f"{arr}\n\n{mix}".strip()))

    return "".join(parts)


# ── Secret Helper event handlers ──────────────────────────────────────────────

def _helper_core(message, messages, helper_song, voice, genre, model_size, instrumental):
    """
    Core generator for Secret Helper calls.
    Yields 5-tuples: (chat_html, messages, helper_state, actions_update, quick_update)
    """
    messages = list(messages) + [("user", message)]
    yield _build(messages), messages, helper_song, gr.update(visible=False), gr.update(visible=False)

    ui_settings = {
        "voice": voice,
        "genre": None if genre == "auto" else genre,
        "bpm":   None,
        "model_size":       model_size,
        "instrumental_only": bool(instrumental),
    }

    res, exc, t = _run_bg(secret_helper.generate, message, ui_settings, helper_song)

    frames = ["▪ ▫ ▫", "▫ ▪ ▫", "▫ ▫ ▪"]
    fi = 0
    messages = messages + [("bot", _status(f"SECRET HELPER IS THINKING {frames[0]}"))]
    yield _build(messages), messages, helper_song, gr.update(visible=False), gr.update(visible=False)

    while t.is_alive():
        time.sleep(1.0)
        fi = (fi + 1) % 3
        messages[-1] = ("bot", _status(f"SECRET HELPER IS THINKING {frames[fi]}"))
        yield _build(messages), messages, helper_song, gr.update(visible=False), gr.update(visible=False)

    t.join()

    if exc[0]:
        messages[-1] = ("bot", _card("⚠ ERROR", str(exc[0])))
        yield _build(messages), messages, helper_song, gr.update(visible=False), gr.update(visible=False)
        return

    result  = res[0]
    card    = _build_helper_card(result)
    messages[-1] = ("bot", card)
    show    = (not result.get("need_clarification", False)
               and bool(result.get("song", {}).get("title")))
    yield (
        _build(messages), messages, result,
        gr.update(visible=show), gr.update(visible=show),
    )


def on_helper_submit(message, messages, helper_song, voice, genre, model_size, instrumental):
    """Triggered by ✦ button — clears prompt_input, streams helper response."""
    if not message.strip():
        yield "", _build(messages), messages, helper_song, gr.update(), gr.update()
        return
    for t in _helper_core(message, messages, helper_song, voice, genre, model_size, instrumental):
        yield ("",) + t


def on_revise_submit(message, messages, helper_song, voice, genre, model_size, instrumental):
    """Triggered by Revise button — clears revise_box, streams updated response."""
    if not message.strip():
        yield "", _build(messages), messages, helper_song, gr.update(), gr.update()
        return
    for t in _helper_core(message, messages, helper_song, voice, genre, model_size, instrumental):
        yield ("",) + t


def on_apply_song(helper_song, messages):
    """Fills voice_drop, genre_drop, and custom_lyrics from the last helper result."""
    if not helper_song:
        return gr.update(), gr.update(), gr.update(), gr.update(), _build(messages), messages

    song      = helper_song.get("song", {})
    new_voice = song.get("voice", "neutral")
    new_genre = song.get("genre", "auto")
    new_lyrics = helper_song.get("lyrics", {}).get("text", "")

    if new_voice not in VOICE_OPTIONS:
        new_voice = "neutral"
    if new_genre not in GENRE_OPTIONS:
        new_genre = "auto"

    title = song.get("title", "untitled")
    bpm   = song.get("bpm", "?")
    note  = f"✓ Applied: {title} · {new_genre} · {bpm} bpm — lyrics loaded into Custom Lyrics box"
    messages = list(messages) + [("bot", _status(note))]

    return (
        gr.update(value=new_voice),
        gr.update(value=new_genre),
        gr.update(value=new_lyrics),
        gr.update(open=True),          # auto-open the lyrics accordion
        _build(messages),
        messages,
    )


def _make_regen(part: str):
    """Factory: returns a handler that regenerates a specific song section."""
    def _handler(messages, helper_song, voice, genre, model_size, instrumental):
        msg = (
            f"Regenerate only the {part}. Keep genre, BPM, voice, and overall story arc. "
            "Make it more specific, powerful, and free of clichés."
        )
        for t in _helper_core(msg, messages, helper_song, voice, genre, model_size, instrumental):
            yield t   # 5-tuple (no prompt clearing)
    return _handler


# ── Generation (streaming) ────────────────────────────────────────────────────

def _run_bg(fn, *args):
    """Run fn(*args) in a daemon thread. Returns (result_box, exc_box, thread)."""
    result, exc = [None], [None]
    def _worker():
        try:
            result[0] = fn(*args)
        except Exception as e:
            exc[0] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return result, exc, t



def on_submit(message, messages, voice, model_size, music_only, genre1, custom_lyrics=""):
    """Streaming generator — yields (cleared_input, chat_html, messages_state, audio_path)."""
    if not message.strip():
        yield "", _build(messages), messages, None
        return

    # Immediately clear input + show user bubble
    messages = list(messages) + [("user", message)]
    yield "", _build(messages), messages, None

    # Parse
    parsed = prompt_parser.parse(message, genre1)
    title  = parsed["theme"][:35].title()
    genre  = parsed["genre"]
    mood   = parsed["mood"]
    style  = f"{genre}, {mood}, {parsed['bpm']} bpm"

    # Warn if Ollama is not ready
    if not _OL_ONLINE or not _OL_MODEL:
        warn = _card("⚠ OLLAMA WARNING", _OL_MSG + "\nLyrics will use the fallback template instead.")
        messages = messages + [("bot", warn)]
        yield "", _build(messages), messages, None

    # Step 1 — crafting lyrics (show spinner while Ollama runs)
    bot = _status(f'CRAFTING LYRICS FOR &ldquo;{title.upper()}&rdquo;...')
    messages = messages + [("bot", bot)]
    yield "", _build(messages), messages, None

    lyrics_text = ""
    if not music_only:
        if custom_lyrics and custom_lyrics.strip():
            # Use the lyrics the user wrote / pasted / applied from helper
            lyrics_text = custom_lyrics.strip()
            messages[-1] = ("bot", _status("USING YOUR LYRICS ✓"))
            yield "", _build(messages), messages, None
        else:
            res, exc, t = _run_bg(lyrics_gen.generate, parsed["theme"], genre, mood)
            frames = ["▪ ▫ ▫", "▫ ▪ ▫", "▫ ▫ ▪"]
            fi = 0
            while t.is_alive():
                time.sleep(1.0)
                fi = (fi + 1) % len(frames)
                messages[-1] = ("bot", _status(f'CRAFTING LYRICS {frames[fi]}'))
                yield "", _build(messages), messages, None
            t.join()
            lyrics_text = res[0] or ""

    # Step 2 — show SOUND + LYRICS cards
    bot  = _status(f'CREATING &ldquo;{title.upper()}&rdquo;... &rsaquo;')
    bot += _card("🔊 SOUND", style)
    if not music_only and lyrics_text:
        bot += _card("📝 LYRICS", lyrics_text)
    messages[-1] = ("bot", bot)
    yield "", _build(messages), messages, None

    # Step 3 — generate music in background, animate 20 %
    res_m, exc_m, t_m = _run_bg(music_gen.generate, parsed["music_prompt"], 30, model_size)
    frames = ["▪ ▫ ▫", "▫ ▪ ▫", "▫ ▫ ▪"]
    fi = 0
    while t_m.is_alive():
        time.sleep(1.5)
        fi = (fi + 1) % len(frames)
        messages[-1] = ("bot", bot + _prog(20, f"Generating beat {frames[fi]}", style[:55]))
        yield "", _build(messages), messages, None
    t_m.join()
    if exc_m[0]:
        raise exc_m[0]
    instrumental, instr_sr = res_m[0]

    if music_only:
        messages[-1] = ("bot", bot + _prog(90, f"Saving {title}...", style[:55]))
        yield "", _build(messages), messages, None

        path = f"output/instrumental_{int(time.time())}.wav"
        path = mixer.save_instrumental(instrumental, instr_sr, path)

        narr = MOOD_NARRATIONS.get(mood, "Here's your track.")
        sugg = SUGGESTIONS.get(genre, SUGGESTIONS["default"])
        final = bot + f'<div class="mas-narration">{narr}</div>' + _chips(sugg[:4])
        messages[-1] = ("bot", final)
        hist.add({"prompt": message, "genre": genre, "mood": mood,
                  "duration": 30, "voice": voice, "path": path, "lyrics": ""})
        yield "", _build(messages), messages, path
        return

    # Step 4 — vocals in background, animate 60 %
    res_v, exc_v, t_v = _run_bg(vocal_gen.generate, lyrics_text, voice)
    fi = 0
    while t_v.is_alive():
        time.sleep(1.5)
        fi = (fi + 1) % len(frames)
        messages[-1] = ("bot", bot + _prog(60, f"Recording vocals {frames[fi]}", style[:55]))
        yield "", _build(messages), messages, None
    t_v.join()
    if exc_v[0]:
        raise exc_v[0]
    vocals, vocal_sr = res_v[0]

    # Step 5 — mixing
    messages[-1] = ("bot", bot + _prog(90, f"Mixing {title}...", style[:55]))
    yield "", _build(messages), messages, None

    path = f"output/song_{int(time.time())}.wav"
    path = mixer.mix(instrumental, instr_sr, vocals, vocal_sr, path)

    # Final — narration + suggestion chips
    narr = MOOD_NARRATIONS.get(mood, "Here's your track.")
    sugg = SUGGESTIONS.get(genre, SUGGESTIONS["default"])
    final = bot + f'<div class="mas-narration">{narr}</div>' + _chips(sugg[:4])
    messages[-1] = ("bot", final)

    hist.add({"prompt": message, "genre": genre, "mood": mood,
              "duration": 30, "voice": voice, "path": path, "lyrics": lyrics_text})
    yield "", _build(messages), messages, path


# ── FastAPI app (owns port 7860; Gradio mounts onto it) ───────────────────────

class _PromptIn(BaseModel):
    prompt: str

api = FastAPI()

@api.post("/api/ai")
def _ai(body: _PromptIn):
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.OLLAMA_MODEL, "prompt": body.prompt.strip(), "stream": False},
        timeout=90,
    )
    resp.raise_for_status()
    return {"response": resp.json()["response"]}


class _HelperIn(BaseModel):
    user_message: str
    ui_settings:  dict = {}
    current_song: dict = {}

@api.post("/api/secret-helper")
def _secret_helper_route(body: _HelperIn):
    return secret_helper.generate(
        body.user_message,
        body.ui_settings,
        body.current_song or None,
    )


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Secret Helper") as demo:

    # Inject CSS + JS via HTML — works in Gradio 6 with mount_gradio_app
    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(f"<script>{JS_SCROLL}</script>")

    messages_state = gr.State([])
    helper_state   = gr.State(None)

    chat_display = gr.HTML(value=HERO_HTML, elem_id="chat-display")
    audio_out    = gr.Audio(type="filepath", label="Your Song")

    # ── Secret Helper action rows (hidden until helper responds) ──
    with gr.Row(visible=False, elem_id="helper-actions") as helper_actions:
        apply_btn  = gr.Button("✓ Apply to Song", elem_id="apply-btn",  scale=1)
        revise_box = gr.Textbox(placeholder="What to change?", show_label=False,
                                lines=1, scale=4, elem_id="revise-box")
        revise_btn = gr.Button("↩ Revise", elem_id="revise-btn", scale=1)

    with gr.Row(visible=False, elem_id="quick-actions") as quick_actions_row:
        regen_hook   = gr.Button("↺ Hook",    scale=1, size="sm", elem_id="regen-hook")
        regen_verse1 = gr.Button("↺ Verse 1", scale=1, size="sm", elem_id="regen-v1")
        regen_verse2 = gr.Button("↺ Verse 2", scale=1, size="sm", elem_id="regen-v2")
        regen_bridge = gr.Button("↺ Bridge",  scale=1, size="sm", elem_id="regen-bridge")
        regen_sound  = gr.Button("↺ Sound",   scale=1, size="sm", elem_id="regen-sound")

    with gr.Column(elem_id="bottom-area"):
        with gr.Accordion("✏  Custom Lyrics", open=False, elem_id="lyrics-accordion") as lyrics_accordion:
            custom_lyrics = gr.Textbox(
                placeholder="Paste your own lyrics here — leave empty to auto-generate...",
                lines=6,
                show_label=False,
                elem_id="custom-lyrics-box",
            )

        with gr.Row():
            voice_drop = gr.Dropdown(VOICE_OPTIONS, value="neutral", label="Voice", scale=1)
            genre_drop = gr.Dropdown(GENRE_OPTIONS, value="auto",    label="Genre", scale=1)
        with gr.Row():
            model_radio = gr.Radio(MODEL_OPTIONS, value="small", label="Model", scale=3)
            instr_cb    = gr.Checkbox(label="Instrumental only", value=False, scale=1)

        with gr.Row(elem_id="input-bar"):
            prompt_input = gr.Textbox(
                placeholder="Ask Secret Helper...",
                lines=2,
                show_label=False,
                elem_id="prompt-box",
                scale=5,
            )
            helper_btn = gr.Button("✦", elem_id="helper-btn", scale=0, min_width=46)
            send_btn   = gr.Button("▶", elem_id="send-btn",  scale=0, min_width=46)

        gr.HTML(_OLLAMA_BANNER)
        gr.HTML('<div id="disclaimer-text">Secret Helper is in beta and can make mistakes.</div>')

    # ── Song generation events ──
    _inputs  = [prompt_input, messages_state, voice_drop, model_radio, instr_cb, genre_drop, custom_lyrics]
    _outputs = [prompt_input, chat_display, messages_state, audio_out]

    send_btn.click(fn=on_submit, inputs=_inputs, outputs=_outputs)
    prompt_input.submit(fn=on_submit, inputs=_inputs, outputs=_outputs)

    # ── Secret Helper events ──
    _h_inputs  = [prompt_input, messages_state, helper_state,
                  voice_drop, genre_drop, model_radio, instr_cb]
    _h_outputs = [prompt_input, chat_display, messages_state,
                  helper_state, helper_actions, quick_actions_row]

    helper_btn.click(fn=on_helper_submit, inputs=_h_inputs, outputs=_h_outputs)

    # Revise button (clears revise_box instead of prompt_input)
    _r_inputs  = [revise_box, messages_state, helper_state,
                  voice_drop, genre_drop, model_radio, instr_cb]
    _r_outputs = [revise_box, chat_display, messages_state,
                  helper_state, helper_actions, quick_actions_row]

    revise_btn.click(fn=on_revise_submit, inputs=_r_inputs, outputs=_r_outputs)

    # Apply to Song (also fills custom_lyrics box and opens accordion)
    apply_btn.click(
        fn=on_apply_song,
        inputs=[helper_state, messages_state],
        outputs=[voice_drop, genre_drop, custom_lyrics, lyrics_accordion, chat_display, messages_state],
    )

    # Quick regen actions (5-tuple outputs — no input box to clear)
    _qr_inputs  = [messages_state, helper_state, voice_drop, genre_drop, model_radio, instr_cb]
    _qr_outputs = [chat_display, messages_state, helper_state, helper_actions, quick_actions_row]

    regen_hook.click(fn=_make_regen("Hook"),    inputs=_qr_inputs, outputs=_qr_outputs)
    regen_verse1.click(fn=_make_regen("Verse 1"), inputs=_qr_inputs, outputs=_qr_outputs)
    regen_verse2.click(fn=_make_regen("Verse 2"), inputs=_qr_inputs, outputs=_qr_outputs)
    regen_bridge.click(fn=_make_regen("Bridge"),  inputs=_qr_inputs, outputs=_qr_outputs)
    regen_sound.click(fn=_make_regen("Sound description"), inputs=_qr_inputs, outputs=_qr_outputs)


# Module-level so uvicorn --reload can find "app:app"
demo.queue()
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=(port == 7860))
