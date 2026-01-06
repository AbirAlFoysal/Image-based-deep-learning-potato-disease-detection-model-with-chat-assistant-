from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        # Load disease detection models on app startup
        from .disease_detection_service import load_models
        load_models()
