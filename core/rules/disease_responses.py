"""
Disease response rules for Dr. Pato AI Assistant.
Contains functions for generating disease-specific responses.
"""


def get_disease_response(disease_type):
    """
    Generate a conversational response for a specific disease type.
    
    Args:
        disease_type (str): The tuber issue detected (e.g., "Brown Rot", "Healthy")
    
    Returns:
        str: A contextual response about the disease type
    """
    responses = {
        "Blackspot Bruising": "I can see signs of blackspot bruising in the tuber. This usually comes from impact damage during harvest or handling. Would you like tips to reduce bruising and storage losses?",
        "Healthy": "Good news, this tuber looks healthy. I don't see obvious disease symptoms in the uploaded image. Would you like some storage or handling tips to help keep it in good condition?",
        "Brown Rot": "This looks consistent with brown rot, a serious bacterial problem often linked to Ralstonia solanacearum. Would you like help with sanitation, seed selection, and field management steps?",
        "Dry Rot": "I can see signs that fit dry rot, which is commonly associated with Fusarium species during storage. Would you like suggestions for storage hygiene and wound management?",
        "Soft Rot": "This resembles soft rot, which is often caused by bacteria such as Pectobacterium spp. and can spread quickly in wet conditions. Would you like advice on handling, drying, and storage prevention?"
    }
    
    return responses.get(disease_type, f"I've detected {disease_type} in the tuber image. Would you like more information about this condition?")


def get_default_response(disease, specific_info=""):
    """
    Generate a default response template for disease detection.
    
    Args:
        disease (str): The detected disease name
        specific_info (str): Additional specific information about the disease (optional)
    
    Returns:
        str: A formatted default response
    """
    if specific_info:
        return f"I've analyzed your potato tuber image. The detected condition is: {disease}. {specific_info} What would you like to know more about - treatment options, prevention, or something else related to potato health?"
    else:
        return f"I've analyzed your potato tuber image. The detected condition is: {disease}. What would you like to know more about - treatment options, prevention, or something else related to potato health?"
