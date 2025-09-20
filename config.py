import os

class Config:
    """
    Configuration settings for the PharmAgent application.
    """
    # General settings
    APP_NAME = "PharmAgent"

    # Deployment settings
    DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV", "local")  # 'local' or 'aws'

    # AWS settings (if applicable)
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "pharmagent-results")