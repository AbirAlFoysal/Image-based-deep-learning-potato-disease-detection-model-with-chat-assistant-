import os
import textwrap
from groq import Groq
from .rules.potato_rules import SYSTEM_PROMPT, DISEASE_RESPONSES, DEFAULT_RESPONSE

class DrPatoAssistant:

    def __init__(self, api_key=None):
        self.api_key = api_key or self._get_api_key()
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

        self.system_prompt = SYSTEM_PROMPT

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _get_api_key(self):
        return os.environ.get('GROQ_API_KEY')

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
        prompt = f"""As Dr. Pato, analyze these potato symptoms and provide a diagnosis:

Symptoms: {symptoms}

Provide:
1. Likely disease(s) with scientific names
2. Primary symptoms match
3. Recommended laboratory tests for confirmation
4. Immediate management steps
5. Prevention for next season"""

        return self.chat(prompt)

    def list_common_diseases(self):
        return self.chat("List the 10 most common potato diseases worldwide with their scientific names and primary symptoms.")

    def get_treatment(self, disease_name):
        return self.chat(f"What are the most effective treatments for {disease_name} in potatoes? Include organic and chemical options.")

    def handle_disease_detection(self, disease_name, image_type='leaf'):
        if disease_name.startswith("Error"):
            return f"There was an error analyzing the image: {disease_name}. Please try again or describe the symptoms manually."
        
        if image_type == 'tuber' and disease_name.startswith('Potato___'):
            disease_name = disease_name.replace('Potato___', '').replace('_', ' ').title()
        
        prompt = f"""I analyzed a potato {image_type} image and detected: {disease_name}

Please respond in a friendly, conversational way as Dr. Pato. Keep it brief and natural. Explain what this means for the potato plant in simple terms, then ask if they'd like to know about remedies, prevention, or anything else.

Example style: "Oh, I see some signs of [disease] here. This usually happens when... Would you like me to suggest some treatment options or tell you how to prevent it in the future?"
Be helpful and engaging, not like a textbook.""" 
        
        return self.chat(prompt)