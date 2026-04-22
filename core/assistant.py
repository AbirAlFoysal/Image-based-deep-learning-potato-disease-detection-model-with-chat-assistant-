import os
import textwrap
try:
    from groq import Groq
except ImportError:
    Groq = None
from .config import Config
from .rules import (
    get_system_prompt,
    get_diagnosis_prompt,
    get_treatment_prompt,
    get_common_diseases_prompt,
    get_disease_detection_prompt
)

class DrPatoAssistant:
    FALLBACK_MODELS = (
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    )

    def __init__(self, api_key=None):
        self.api_key = api_key or self._get_api_key()
        if self.api_key and Groq is not None:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

        self.system_prompt = get_system_prompt()
        self.model = Config.get_groq_model()

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _get_api_key(self):
        return Config.get_groq_api_key_safe()

    def _format_response(self, text, width=70):
        return textwrap.fill(text, width=width)

    def _create_chat_completion(self, model_name):
        return self.client.chat.completions.create(
            messages=self.conversation_history,
            model=model_name,
            temperature=0.7,
            max_tokens=300,
            top_p=0.95,
        )

    def _get_model_candidates(self):
        candidates = [self.model]
        for fallback_model in self.FALLBACK_MODELS:
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return candidates

    def chat(self, user_input):
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            if not self.client:
                if Groq is None:
                    return "AI service not configured. Please install the groq package."
                return "AI service not configured. Please set GROQ_API_KEY."

            last_error = None
            response = None
            for model_name in self._get_model_candidates():
                try:
                    response = self._create_chat_completion(model_name)
                    self.model = model_name
                    break
                except Exception as model_error:
                    last_error = model_error
                    error_text = str(model_error).lower()
                    if "model_not_found" not in error_text and "does not exist" not in error_text:
                        raise

            if response is None:
                raise last_error

            dr_pato_response = response.choices[0].message.content

            self.conversation_history.append({
                "role": "assistant",
                "content": dr_pato_response
            })

            return dr_pato_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg

    def diagnose_potato(self, symptoms):
        prompt = get_diagnosis_prompt(symptoms)
        return self.chat(prompt)

    def list_common_diseases(self):
        return self.chat(get_common_diseases_prompt())

    def get_treatment(self, disease_name):
        return self.chat(get_treatment_prompt(disease_name))

    def handle_disease_detection(self, disease_name):
        if disease_name.startswith("Error"):
            return "I couldn't analyze that image right now. Please try again or describe the symptoms manually."
        
        prompt = get_disease_detection_prompt(disease_name)
        return self.chat(prompt)
