import os
import textwrap
from groq import Groq
from .config import Config
from .rules import (
    get_system_prompt,
    get_diagnosis_prompt,
    get_treatment_prompt,
    get_common_diseases_prompt,
    get_disease_detection_prompt
)

class DrPatoAssistant:

    def __init__(self, api_key=None):
        self.api_key = api_key or self._get_api_key()
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

        self.system_prompt = get_system_prompt()

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _get_api_key(self):
        return Config.get_groq_api_key_safe()

    def _format_response(self, text, width=70):
        return textwrap.fill(text, width=width)

    def chat(self, user_input):
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            if not self.client:
                return "AI service not configured. Please set GROQ_API_KEY."
            response = self.client.chat.completions.create(
                messages=self.conversation_history,
                model="moonshotai/kimi-k2-instruct",  
                temperature=0.7,
                max_tokens=300,  
                top_p=0.95,
            )

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

    def handle_disease_detection(self, disease_name, image_type='leaf'):
        if disease_name.startswith("Error"):
            return f"There was an error analyzing the image: {disease_name}. Please try again or describe the symptoms manually."
        
        prompt = get_disease_detection_prompt(disease_name, image_type)
        return self.chat(prompt)