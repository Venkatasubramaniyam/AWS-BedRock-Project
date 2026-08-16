import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "amazon.nova-micro-v1:0"
)

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "500")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.3")
)

MAX_HISTORY = int(
    os.getenv("MAX_HISTORY", "10")
)