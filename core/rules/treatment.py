"""
Treatment prompt rule for Dr. Pato AI Assistant.
Generates prompts for requesting treatment information.
"""


def get_treatment_prompt(disease_name):
    """
    Generate a prompt for getting treatment information for a specific disease.
    
    Args:
        disease_name (str): The name of the potato disease
    
    Returns:
        str: A prompt requesting treatment options
    """
    return f"What are the most effective treatments for {disease_name} in potatoes? Include organic and chemical options."
