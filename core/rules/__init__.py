"""
Rules package for Dr. Pato AI Assistant.
Contains all prompt generation functions organized by purpose.

This package provides clean separation of prompt generation logic,
with each type of prompt in its own dedicated module.
"""

# Import all functions to make them available from the package level
from .system_prompt import get_system_prompt
from .disease_responses import get_disease_response, get_default_response
from .diagnosis import get_diagnosis_prompt
from .treatment import get_treatment_prompt
from .common_diseases import get_common_diseases_prompt
from .disease_detection import get_disease_detection_prompt
from .helpers import get_off_topic_response, get_image_analysis_encouragement

# Define what's available when using "from rules import *"
__all__ = [
    'get_system_prompt',
    'get_disease_response',
    'get_default_response',
    'get_diagnosis_prompt',
    'get_treatment_prompt',
    'get_common_diseases_prompt',
    'get_disease_detection_prompt',
    'get_off_topic_response',
    'get_image_analysis_encouragement',
]
