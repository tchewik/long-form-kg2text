import os

WANDB_KEY = os.getenv("WANDB_API_KEY", "")

CLEARML_API_HOST = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST", "https://app.clear.ml")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST", "https://files.clear.ml")
CLEARML_KEY = os.getenv("CLEARML_API_ACCESS_KEY", "")
CLEARML_SECRET = os.getenv("CLEARML_API_SECRET_KEY", "")
