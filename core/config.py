from pathlib import Path
from decouple import config, Csv

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    GROQ_API_KEY = config('GROQ_API_KEY', default='')
    GROQ_MODEL = config('GROQ_MODEL', default='llama-3.1-8b-instant')
    @classmethod
    def get_groq_api_key(cls):
        api_key = cls.GROQ_API_KEY
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Please add it to your .env file in the app directory."
            )
        return api_key
    
    @classmethod
    def get_groq_api_key_safe(cls):
        return cls.GROQ_API_KEY or None

    @classmethod
    def get_groq_model(cls):
        return cls.GROQ_MODEL or 'llama-3.1-8b-instant'
