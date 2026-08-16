from flask import Flask, render_template, request, jsonify
import boto3

app = Flask(__name__)

# Create Bedrock Runtime client
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

# Bedrock model
MODEL_ID = "amazon.nova-micro-v1:0"


@app.route("/")
def home():
    return render_template("index.html") 


@app.route("/ask", methods=["POST"]) 
def ask():

    data = request.get_json() 

    user_question = data.get("question")
	
    if not user_question:
        return jsonify({
            "error": "Question is required"
        }), 400

    try:

        response = bedrock.converse(
            modelId=MODEL_ID,

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": user_question
                        }
                    ]
                }
            ],

            inferenceConfig={
                "maxTokens": 500,
                "temperature": 0.3
            }
        )

        answer = response["output"]["message"]["content"][0]["text"]

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )