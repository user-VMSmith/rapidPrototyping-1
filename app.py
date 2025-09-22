"""Simple Flask front-end for interacting with an LM Studio API."""

from flask import Flask, render_template, request
import requests
import math

# Initialize the Flask application
app = Flask(__name__)

# URL of the LM Studio API we want to communicate with
LM_API_URL = "http://localhost:1234/v1/chat/completions"
EMBEDDINGS_API_URL = "http://localhost:1234/v1/embeddings"
# Authentication token for the API
LM_API_KEY = "lm-studio"

@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and handle interactions with the LM Studio API."""

    # Placeholder variables for the response and user/system prompts
    response_text = ""
    user_input = ""
    system_prompt = "You are a helpful assistant."

    # When the form is submitted we send the input to the language model
    if request.method == "POST":
        # Prompt typed by the user
        user_input = request.form.get("prompt", "")
        # Optional override for the default system prompt
        system_prompt = request.form.get("system", system_prompt)

        # JSON payload describing the chat conversation and parameters
        payload = {
            "model": "nomic-embed-text-v1.5",  # model served by LM Studio
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "max_tokens": 150,  # limit the size of the response
            "temperature": 0.7,
        }

        # HTTP headers including API authentication token
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_API_KEY}",
        }

        try:
            # Send the request to the language model server
            r = requests.post(LM_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # Extract the assistant's reply from the response payload
            response_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            # Surface any errors to the user
            response_text = f"Error: {str(e)}"

    # Render the web page with the model response and current prompts
    return render_template(
        "index.html", response=response_text, prompt=user_input, system=system_prompt
    )

@app.route("/embeddings", methods=["GET", "POST"])
def embeddings():
    similarity = None
    text1 = ""
    text2 = ""

    if request.method == "POST":
        text1 = request.form.get("text1", "")
        text2 = request.form.get("text2", "")

        payload = {
            "model": "nomic-embed-text-v1.5",
            "input": [text1, text2]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_API_KEY}"
        }

        try:
            r = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            embeds = [item["embedding"] for item in data.get("data", [])]
            if len(embeds) >= 2:
                similarity = round(cosine_similarity(embeds[0], embeds[1]), 4)
        except Exception as e:
            similarity = f"Error: {str(e)}"

    return render_template(
        "embeddings.html",
        similarity=similarity,
        text1=text1,
        text2=text2,
    )

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

if __name__ == "__main__":
    app.run(debug=True)
