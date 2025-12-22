import os
import textwrap
from groq import Groq

class DrPatoAssistant:
    """AI Assistant specialized in potato diseases only."""

    def __init__(self, api_key=None):
        self.api_key = api_key or self._get_api_key()
        self.client = Groq(api_key=self.api_key)

        # System prompt to enforce specialization
        self.system_prompt = """You are Dr. Pato, a world-renowned potato pathologist and disease specialist. Your expertise is EXCLUSIVELY in potato diseases, disorders, and health issues.

STRICT RULES YOU MUST FOLLOW:
1. ONLY discuss potato diseases, pathogens, disorders, and related agricultural topics
2. If asked about non-potato topics, respond: "I specialize only in potato diseases. Please ask me about potato blight, scab, wilt, rot, or other potato health issues."
3. Provide accurate, scientific information about potato diseases
4. Include Latin names of pathogens when relevant
5. Offer prevention and treatment advice when appropriate
6. Keep responses focused and professional

EXAMPLES OF APPROPRIATE TOPICS:
- Late blight (Phytophthora infestans)
- Early blight (Alternaria solani)
- Common scab (Streptomyces scabies)
- Blackleg and soft rot (Pectobacterium spp.)
- Potato virus Y (PVY)
- Potato cyst nematodes
- Nutrient deficiencies in potatoes
- Fungicide recommendations for potatoes
- Disease-resistant potato varieties

Remember: You are Dr. Pato. Potato diseases are your life's work."""

        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _get_api_key(self):
        """Get API key - hardcoded"""
        return "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"

    def _format_response(self, text, width=70):
        """Format text for better readability"""
        return textwrap.fill(text, width=width)

    def chat(self, user_input):
        """Process user input and get Dr. Pato's response"""

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            # Get response from Groq using moonshotai/kimi-k2-instruct
            response = self.client.chat.completions.create(
                messages=self.conversation_history,
                model="moonshotai/kimi-k2-instruct",  # Your specified model
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
            )

            dr_pato_response = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": dr_pato_response
            })

            return dr_pato_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg

    def diagnose_potato(self, symptoms):
        """Specialized diagnosis function"""
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
        """Get list of common potato diseases"""
        return self.chat("List the 10 most common potato diseases worldwide with their scientific names and primary symptoms.")

    def get_treatment(self, disease_name):
        """Get treatment for specific disease"""
        return self.chat(f"What are the most effective treatments for {disease_name} in potatoes? Include organic and chemical options.")

    def clear_conversation(self):
        """Reset conversation history"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        return "Conversation cleared. I'm ready to discuss potato diseases!"