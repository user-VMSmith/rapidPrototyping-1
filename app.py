from flask import Flask, render_template, request
import requests

app = Flask(__name__)

LM_API_URL = "http://localhost:1234/v1/chat/completions"
LM_API_KEY = "lm-studio"

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    user_input = ""
    system_prompt = "You are a helpful assistant."

    if request.method == "POST":
        user_input = request.form.get("prompt", "")
        system_prompt = request.form.get("system", system_prompt)

        payload = {
            "model": "openhermes-2.5-mistral-7b",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_input }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_API_KEY}"
        }

        try:
            r = requests.post(LM_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            response_text = data['choices'][0]['message']['content']
        except Exception as e:
            response_text = f"Error: {str(e)}"

    return render_template("index.html", response=response_text, prompt=user_input, system=system_prompt)

if __name__ == "__main__":
    app.run(debug=True)
