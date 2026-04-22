"""
Disease detection prompt rule for Dr. Pato AI Assistant.
Generates conversational prompts for disease detection from image analysis.
"""


def get_disease_detection_prompt(disease_name):
    """
    Generate a conversational prompt for tuber disease detection from image analysis.
    
    Args:
        disease_name (str): The detected disease name
    
    Returns:
        str: A conversational prompt for the AI to respond naturally
    """

    return f"""I analyzed a potato tuber image and detected: {disease_name}

Please respond in a friendly, conversational way as Dr. Pato. Keep it brief and natural. Explain what this means for the potato plant in simple terms, then ask if they'd like to know about remedies, prevention, or anything else.

Example style: "Oh, I see some signs of [disease] here. This usually happens when... Would you like me to suggest some treatment options or tell you how to prevent it in the future?"
Be helpful and engaging, not like a textbook."""
