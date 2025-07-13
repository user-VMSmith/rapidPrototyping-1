"""Simple Flask front-end for interacting with an LM Studio API."""

from flask import Flask, render_template, request
import requests

# Initialize the Flask application
app = Flask(__name__)

# URL of the LM Studio API we want to communicate with
LM_API_URL = "http://localhost:1234/v1/chat/completions"
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
            "model": "openhermes-2.5-mistral-7b",  # model served by LM Studio
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

if __name__ == "__main__":
    app.run(debug=True)
