import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

model_id = "amazon.nova-micro-v1:0"

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": user_input
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 500,
            "temperature": 0
        }
    )

    answer = response["output"]["message"]["content"][0]["text"]

    print("AI:", answer)