import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

model_id = "amazon.nova-micro-v1:0"

response = client.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain Genetic AI in simple terms."
                }
            ]
        }
    ],
    inferenceConfig={
        "maxTokens": 500,
        "temperature": 0.3,
        "topP": 0.9
    }
)

answer = response["output"]["message"]["content"][0]["text"]

print(answer)