import os
from typing import List, Dict, Any

import gradio as gr
from PIL import Image
import google.generativeai as genai

# Gemini setup 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY. Add it in your Space secrets.")

PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are 'Data Science Mentor'—a precise, encouraging AI tutor for data science.
- Explain step-by-step with concise runnable Python when helpful.
- Prefer pandas, numpy, matplotlib, scikit-learn.
- Keep answers accurate and cite exact function/class names.
- For debugging screenshots: identify the exact mistake, explain why it fails, then provide a corrected code block.
- Important: Return the fixed code inside a single fenced block: ```python ... ```
"""

def _pick_model():
    try:
        return genai.GenerativeModel(PREFERRED_MODEL)
    except Exception:
        return genai.GenerativeModel("gemini-1.5-flash")

# Helpers 
def _history_to_gemini(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gem_history = []
    for turn in history:
        role = turn["role"]
        parts = []
        for c in turn["content"]:
            if isinstance(c, str):
                parts.append({"text": c})
            elif isinstance(c, Image.Image):
                parts.append(c)
        gem_history.append({"role": role, "parts": parts})
    return gem_history

def chat_infer(user_text: str, image: Image.Image, chat_state):
    if chat_state is None:
        chat_state = []

    user_parts = [user_text.strip()] if user_text else []
    if image is not None:
        user_parts.append(image)
    if not user_parts:
        return chat_state, "Please enter a question or upload an image."

    full = [{"role": "user", "content": [SYSTEM_PROMPT]}]
    full.extend(chat_state)
    full.append({"role": "user", "content": user_parts})

    try:
        resp = _pick_model().generate_content(_history_to_gemini(full))
        text = resp.text or "(No text response)"
    except Exception as e:
        text = f"⚠️ Gemini error: {e}"

    chat_state.append({"role": "user", "content": user_parts})
    chat_state.append({"role": "model", "content": [text]})
    return chat_state, text

def analyze_code_screenshot(img: Image.Image, extra_notes: str):
    if img is None:
        return "Please upload a screenshot of the error/code."

    prompt = (
        "You are a Python debugging assistant. The user uploaded a screenshot that likely shows "
        "a traceback, error message, or code snippet. Do ALL of the following in order:\n"
        "1) **Pinpoint the exact issue**: reference the specific line/construct visible in the image.\n"
        "2) **Explain why it fails** in 2–5 sentences.\n"
        "3) **Provide a fully corrected code block** that runs end-to-end. Put ONLY ONE fenced block and mark it as python.\n"
        "4) If assumptions are needed, state them briefly above the code.\n"
        "Keep the answer compact and practical."
    )
    parts = [{"text": prompt}]
    if extra_notes and extra_notes.strip():
        parts.append({"text": f"User notes/context:\n{extra_notes.strip()}"} )
    parts.append(img)

    try:
        resp = _pick_model().generate_content(parts)
        return resp.text or "I couldn't extract enough detail—try a clearer screenshot."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

#UI
with gr.Blocks(title="Data Science Mentor — Gemini 2.0 Flash") as demo:
    gr.Markdown(
        "# Data Science Mentor — Gemini 2.0 Flash\n"
        "Ask questions, analyze error screenshots, or look up library docs."
    )

    # Chat tab
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=420, type="messages", avatar_images=(None, None))
        chat_state = gr.State([])
        with gr.Row():
            txt = gr.Textbox(
                label="Ask about DS concepts, libraries, algorithms…",
                placeholder="e.g., Explain cross-validation with scikit-learn example",
                scale=4,
            )
            img = gr.Image(label="(Optional) Image", type="pil", height=180)
        with gr.Row():
            send = gr.Button("Send", variant="primary")

        def _ui_send(user_text, image, state):
            state, answer = chat_infer(user_text, image, state)
            messages = []
            for turn in state:
                role = "user" if turn["role"] == "user" else "assistant"
                parts = []
                for c in turn["content"]:
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, Image.Image):
                        parts.append("[Image attached]")
                messages.append({"role": role, "content": "\n".join(parts)})
            return messages, state, gr.update(value=None), None

        send.click(_ui_send, [txt, img, chat_state], [chatbot, chat_state, txt, img])

    # Debug from Screenshot
    with gr.Tab("Debug from Screenshot"):
        gr.Markdown(
            "Upload a screenshot of your **Python error or code**. "
            "I’ll point to the bug, explain it, and return a corrected script."
        )
        with gr.Row():
            ss = gr.Image(type="pil", label="Drop your error/code screenshot here", height=360)
            notes = gr.Textbox(
                label="(Optional) Context (Python version, library versions, what you expected, etc.)",
                lines=6,
                placeholder="e.g., Python 3.11, pandas 2.2; expected df.merge to work with column 'id'",
            )
        fix_btn = gr.Button("Analyze & Fix", variant="primary")
        fix_md = gr.Markdown()
        fix_btn.click(analyze_code_screenshot, [ss, notes], fix_md)

    # Docs Lookup
    with gr.Tab("Docs Lookup"):
        with gr.Row():
            mod_tb = gr.Textbox(label="Module (e.g., pandas, numpy, sklearn.metrics)", value="pandas")
            sym_tb = gr.Textbox(label="Symbol path (optional, e.g., DataFrame.merge)", value="DataFrame.merge")
        fetch = gr.Button("Fetch Docs")
        docs_out = gr.Markdown()
        def fetch_docs(module_name: str, symbol_path: str):
            import importlib, inspect
            module_name = (module_name or "").strip()
            symbol_path = (symbol_path or "").strip()
            if not module_name:
                return "Enter a module name like 'pandas' or 'sklearn.metrics'."
            try:
                mod = importlib.import_module(module_name)
            except Exception as e:
                return f"⚠️ Could not import '{module_name}': {e}"
            obj = mod
            if symbol_path:
                for p in symbol_path.split("."):
                    if not hasattr(obj, p):
                        return f"⚠️ '{symbol_path}' not found in {module_name}."
                    obj = getattr(obj, p)
            try:
                sig = ""
                if callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "(signature unavailable)"
                doc = inspect.getdoc(obj) or "(no docstring found)"
                header = f"{module_name}.{symbol_path}".strip(".")
                return f"### {header}\n\n**Signature:** `{sig}`\n\n**Docs:**\n{doc}"
            except Exception as e:
                return f"⚠️ Failed to fetch docs: {e}"

        fetch.click(fetch_docs, [mod_tb, sym_tb], docs_out)

    gr.Markdown(
        "_Note:_ Gemini can read most screenshots directly. If the image is blurry, try re-uploading a clearer crop."
    )

if __name__ == "__main__":
    demo.launch()
