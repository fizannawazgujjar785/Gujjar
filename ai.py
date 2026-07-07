import os

from config import GEMINI_API_KEY, OPENAI_API_KEY

try:
    from google import genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if genai and GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

class AIAgent:
    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": "You are Jarvis, a helpful digital assistant for Fizan. Keep responses clear, polite, and useful."
            }
        ]

    def ask(self, prompt):
        if not prompt or not prompt.strip():
            return "Please say something so I can help."

        self.history.append({"role": "user", "content": prompt.strip()})
        response_text = self._generate_response()
        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def _generate_response(self):
        if openai and OPENAI_API_KEY:
            return self._ask_openai(self.history)

        if gemini_client:
            return self._ask_gemini(self.history[-1]["content"])

        return "AI is not configured. Set OPENAI_API_KEY or GEMINI_API_KEY in config or environment."

    @staticmethod
    def _ask_openai(history):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=history,
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return f"OpenAI error: {exc}"

    @staticmethod
    def _ask_gemini(prompt):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return str(response.text).strip()
        except Exception as exc:
            return f"Gemini error: {exc}"



       