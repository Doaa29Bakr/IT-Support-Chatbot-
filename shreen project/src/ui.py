"""
src/ui.py
─────────────────────────────────────────────────────────────────
Responsibility: Gradio chat interface for the IT Support Chatbot.

Features:
  • Polished dark-themed chat UI with custom CSS
  • Per-browser-tab session isolation (no memory bleed between users)
  • Streaming-style feel with instant response rendering
  • Sidebar with quick-issue buttons for common problems
  • Clear chat / reset session button
  • Status indicator showing retrieval is active
─────────────────────────────────────────────────────────────────
"""

import uuid
import gradio as gr
from src.chain import build_chain, get_session_history

# ── Build the chain ONCE at startup (expensive — don't rebuild per request) ──
print("[UI] Building RAG chain...")
chain = build_chain()
print("[UI] Chain ready.")


# ── Core chat function ────────────────────────────────────────────

def chat(user_message: str, history: list, session_id: str) -> tuple:
    if not user_message.strip():
        return "", history

    response = chain.invoke(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    # ✅ Gradio 6 message format
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": response})
    return "", history



def clear_chat(session_id: str) -> tuple:
    """
    Wipe the Gradio display AND the LangChain memory for this session.
    A new session_id is generated so the next conversation starts fresh.
    """
    # Clear LangChain memory for this session
    history_obj = get_session_history(session_id)
    history_obj.clear()
    new_session_id = str(uuid.uuid4())
    return [], new_session_id 


def use_quick_prompt(prompt: str, history: list, session_id: str) -> tuple:
    """
    Triggered when a Quick Issue button is clicked.
    Injects the preset prompt as if the user typed it.
    """
    return chat(prompt, history, session_id)


# ── Custom CSS ────────────────────────────────────────────────────
# Dark professional theme matching an IT/tech support aesthetic

CSS = """
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #f0f4f8;
    --bg-secondary:  #ffffff;
    --bg-card:       #e8edf2;
    --bg-input:      #ffffff;
    --accent-blue:   #0066cc;
    --accent-green:  #28a745;
    --accent-orange: #e67e00;
    --border:        #c8d0da;
    --text-primary:  #1a1f2e;
    --text-muted:    #5a6478;
    --user-bubble:   #d0e8ff;
    --bot-bubble:    #ffffff;
    --font-main:     'IBM Plex Sans', sans-serif;
    --font-mono:     'IBM Plex Mono', monospace;
}

/* ── Global reset ── */
* { box-sizing: border-box; }

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border-bottom: 1px solid var(--border);
    border-radius: 12px 12px 0 0;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0;
}

.header-icon {
    font-size: 2rem;
    line-height: 1;
}

.header-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: var(--font-mono);
    letter-spacing: -0.5px;
    margin: 0;
}

.header-subtitle {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 2px 0 0 0;
    font-weight: 300;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
    animation: pulse 2s infinite;
    margin-left: auto;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chatbot window ── */
#chatbot-window {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 !important;
    min-height: 480px;
}

/* ── Message bubbles ── */
.message.user {
    background: var(--user-bubble) !important;
    border: 1px solid #2d4a6e !important;
    border-radius: 12px 12px 2px 12px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

.message.bot {
    background: var(--bot-bubble) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px 12px 12px 2px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}

/* ── Input row ── */
#input-row {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-top: none;
    padding: 14px 16px;
    border-radius: 0 0 12px 12px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

#user-input textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    resize: none !important;
}

#user-input textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
    outline: none !important;
}

#user-input textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Buttons ── */
#send-btn {
    background: var(--accent-blue) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    min-width: 80px;
}

#send-btn:hover { opacity: 0.85 !important; }

#clear-btn {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    border-color: #f85149 !important;
    color: #f85149 !important;
}

/* ── Quick issues panel ── */
.quick-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-top: 16px;
}

.quick-panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-family: var(--font-mono);
    margin-bottom: 10px;
}

.quick-btn button {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    padding: 7px 10px !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}

.quick-btn button:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: rgba(88,166,255,0.08) !important;
}

/* ── Info panel ── */
.info-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.6;
}

.info-panel strong {
    color: var(--accent-orange);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Session badge ── */
.session-badge {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    margin-top: 8px;
    display: inline-block;
}

/* ── Hide Gradio footer & branding ── */
footer { display: none !important; }
.built-with { display: none !important; }
"""


# ── Quick issue presets ───────────────────────────────────────────

QUICK_ISSUES = [
    ("🐌  Laptop very slow",          "My laptop is extremely slow after logging in."),
    ("📶  Can't connect to Wi-Fi",    "I cannot connect to the Wi-Fi network."),
    ("💙  Blue screen crash",         "My computer randomly shows a blue screen and restarts."),
    ("🔑  Forgot my password",        "I forgot my Windows password and I'm locked out."),
    ("🖨️  Printer not working",       "My printer is offline and won't print anything."),
    ("🔇  No sound / audio issue",    "There is no sound coming from my laptop."),
    ("🔋  Battery draining fast",     "My laptop battery dies way too fast."),
    ("📁  File Explorer crashing",    "File Explorer crashes every time I open a folder."),
]


# ── Gradio UI layout ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Alex — IT Support") as demo:

        # ── Hidden session state (one UUID per browser tab) ───────
        session_id = gr.State(value=lambda: str(uuid.uuid4()))

        # ── Header ───────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <div class="header-icon">🖥️</div>
            <div>
                <p class="header-title">Alex · IT Support</p>
                <p class="header-subtitle">
                    Context-aware assistant · 99 issues in knowledge base
                </p>
            </div>
            <div class="status-dot" title="RAG engine active"></div>
        </div>
        """)

        # ── Main layout: chat (left) + sidebar (right) ────────────
        with gr.Row(equal_height=False):

            # ── Chat column ───────────────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot-window",
                    label="",
                    show_label=False,
                    height=480,
                    avatar_images=(
                       None,
                     "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=alex&backgroundColor=1c2128",
                    ),
                )

                with gr.Row(elem_id="input-row"):
                    user_input = gr.Textbox(
                        placeholder="Describe your IT issue... (Press Enter to send)",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        elem_id="user-input",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button(
                        "Send ↑",
                        elem_id="send-btn",
                        scale=1,
                        variant="primary",
                    )
                    clear_btn = gr.Button(
                        "🗑 Clear",
                        elem_id="clear-btn",
                        scale=1,
                    )

            # ── Sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):

                gr.HTML('<div class="quick-panel-title">⚡ Quick Issues</div>')

                quick_btns = []
                for label, prompt in QUICK_ISSUES:
                    btn = gr.Button(label, elem_classes=["quick-btn"])
                    quick_btns.append((btn, prompt))

                gr.HTML("""
                <div class="info-panel">
                    <strong>How it works</strong><br><br>
                    Alex searches a curated IT knowledge base using
                    semantic similarity — not just keywords.<br><br>
                    <strong>Memory</strong><br><br>
                    Follow-up questions like <em>"how do I check that?"</em>
                    are understood in context of your conversation.<br><br>
                    <strong>Scope</strong><br><br>
                    Windows OS · Network · Hardware · Performance · Auth
                </div>
                """)

        # ── Welcome message (shown on load) ──────────────────────
        demo.load(
       fn=lambda: [
        {
            "role": "assistant",
            "content": (
                "👋 Hi there! I'm **Alex**, your IT Support Specialist.\n\n"
                "I'm here to help you troubleshoot technical issues — just describe "
                "what's going wrong and I'll walk you through it step by step.\n\n"
                "You can also tap one of the **Quick Issues** on the right to get started instantly!"
            )
        }
    ],
    outputs=[chatbot],
)

        # ── Event wiring ──────────────────────────────────────────
        send_btn.click(
            fn=chat,
            inputs=[user_input, chatbot, session_id],
            outputs=[user_input, chatbot],
        )

        user_input.submit(
            fn=chat,
            inputs=[user_input, chatbot, session_id],
            outputs=[user_input, chatbot],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[session_id],
            outputs=[chatbot, session_id],
        )

        for btn, prompt in quick_btns:
            btn.click(
                fn=use_quick_prompt,
                inputs=[
                    gr.State(value=prompt),
                    chatbot,
                    session_id,
                ],
                outputs=[user_input, chatbot],
            )

    return demo