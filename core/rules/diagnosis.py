"""
Diagnosis prompt rule for Dr. Pato AI Assistant.
Generates prompts for diagnosing potato diseases based on symptoms.
"""


def get_diagnosis_prompt(symptoms):
    """
    Generate a prompt for diagnosing potato diseases based on symptoms.
    
    Args:
        symptoms (str): The symptoms observed in the potato plant
    
    Returns:
        str: A detailed diagnostic prompt for the AI model
    """
    return f"""As Dr. Pato, analyze these potato symptoms and provide a diagnosis:

Symptoms: {symptoms}

Provide:
1. Likely disease(s) with scientific names
2. Primary symptoms match
3. Recommended laboratory tests for confirmation
4. Immediate management steps
5. Prevention for next season"""
