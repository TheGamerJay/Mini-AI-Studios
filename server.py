"""
Flask backend — /api/ai endpoint that proxies prompts to Ollama.
Run:  python server.py
Port: 5000  (Gradio runs separately on 7860)
"""
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


@app.route("/api/ai", methods=["POST"])
def ai():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=90,
    )
    resp.raise_for_status()
    return jsonify({"response": resp.json()["response"]})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
