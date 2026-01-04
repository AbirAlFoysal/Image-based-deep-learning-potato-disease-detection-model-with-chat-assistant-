"""
Disease response rules for Dr. Pato AI Assistant.
Contains functions for generating disease-specific responses.
"""


def get_disease_response(disease_type):
    """
    Generate a conversational response for a specific disease type.
    
    Args:
        disease_type (str): The type of disease detected (e.g., "Bacteria", "Fungi", "Healthy")
    
    Returns:
        str: A contextual response about the disease type
    """
    responses = {
        "Bacteria": "I detected bacterial infection in your potato leaf. Bacterial diseases in potatoes are often caused by pathogens like Pectobacterium spp. or Dickeya spp. These can lead to soft rot and blackleg. Would you like to know about treatment options or prevention methods?",
        "Fungi": "The leaf shows fungal infection. Common fungal diseases include late blight (Phytophthora infestans) or early blight (Alternaria solani). Fungal pathogens thrive in humid conditions. Can I help you with fungicide recommendations or cultural control practices?",
        "Healthy": "Great news! Your potato leaf appears healthy with no visible signs of disease. Continue with good agricultural practices to maintain plant health. Is there anything specific about potato care you'd like to discuss?",
        "Nematode": "I see signs of nematode damage. Potato cyst nematodes (Globodera spp.) can cause significant yield losses. These microscopic worms attack roots. Would you like information on resistant varieties or nematode management strategies?",
        "Pest": "The leaf shows pest damage. Various insects like aphids, potato beetles, or mites can affect potatoes. Proper pest management is crucial. Are you interested in organic pest control methods or chemical options?",
        "Phytopthora": "This looks like Phytophthora infection, likely late blight caused by Phytophthora infestans. This is one of the most devastating potato diseases worldwide. Immediate action is needed. Would you like to know about emergency treatment protocols?",
        "Virus": "The leaf exhibits viral symptoms. Common potato viruses include Potato Virus Y (PVY), Potato Leaf Roll Virus (PLRV), or Potato Virus X (PVX). Viruses are often spread by aphids. Do you want information on virus-tested seed potatoes or vector control?"
    }
    
    return responses.get(disease_type, f"I've detected {disease_type} in your potato plant. Would you like more information about this condition?")


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
        return f"I've analyzed your potato leaf image. The detected condition is: {disease}. {specific_info} What would you like to know more about - treatment options, prevention, or something else related to potato health?"
    else:
        return f"I've analyzed your potato leaf image. The detected condition is: {disease}. What would you like to know more about - treatment options, prevention, or something else related to potato health?"
