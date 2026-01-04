"""
Disease detection prompt rule for Dr. Pato AI Assistant.
Generates conversational prompts for disease detection from image analysis.
"""


def get_disease_detection_prompt(disease_name, image_type='leaf'):
    """
    Generate a conversational prompt for disease detection from image analysis.
    
    Args:
        disease_name (str): The detected disease name
        image_type (str): The type of image analyzed ('leaf' or 'tuber')
    
    Returns:
        str: A conversational prompt for the AI to respond naturally
    """

    cleaned_disease = disease_name
    if image_type == 'tuber' and disease_name.startswith('Potato___'):
        cleaned_disease = disease_name.replace('Potato___', '').replace('_', ' ').title()
    
    return f"""I analyzed a potato {image_type} image and detected: {cleaned_disease}

Please respond in a friendly, conversational way as Dr. Pato. Keep it brief and natural. Explain what this means for the potato plant in simple terms, then ask if they'd like to know about remedies, prevention, or anything else.

Example style: "Oh, I see some signs of [disease] here. This usually happens when... Would you like me to suggest some treatment options or tell you how to prevent it in the future?"
Be helpful and engaging, not like a textbook."""
