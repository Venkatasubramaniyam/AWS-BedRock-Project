import boto3
import os
import logging

from flask import Flask, render_template, request, jsonify, session

from config import AWS_REGION, BEDROCK_MODEL_ID, MAX_TOKENS, TEMPERATURE, MAX_HISTORY


app = Flask(__name__)   #creates the Flask application

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-key"
)

# --------------------------------------------------
# Logging
# --------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Bedrock client
# --------------------------------------------------

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a Telecom Operations AI Assistant.

Answer telecom questions accurately and concisely.

Your areas include:
- Telecom BSS and OSS
- NOC operations
- 4G and 5G
- Incident management
- SLA, SLI, SLO and KPI
- Network operations
- Charging and billing
- Network performance metrics
- Network monitoring and troubleshooting
- Network optimization
- Network security and compliance
- Implementation of telecom standards and protocols
- Integration of telecom systems and platforms
- Design and architecture of telecom networks
- Design 5G and 4G networks


For simple definition questions such as:
"What is MTTR?"
"What is SLA?"
"What is KPI?"

give a short answer in 2-4 sentences.

Use a formula or short example when it adds value.

For complex questions, provide more detailed explanations.

Do not invent information.
If you don't know the answer, clearly say so.
"""


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# Chat API
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Invalid request."
            }), 400

        user_question = data.get(
            "question",
            ""
        ).strip()

        if not user_question:

            return jsonify({
                "error": "Please enter a question."
            }), 400

        logger.info(
            "Received question"
        )


        # ------------------------------------------
        # Get existing conversation history
        # ------------------------------------------

        history = session.get(
            "history",
            []
        )


        # ------------------------------------------
        # Add user message
        # ------------------------------------------

        history.append({
            "role": "user",
            "content": [
                {
                    "text": user_question
                }
            ]
        })


        # ------------------------------------------
        # Limit conversation history
        # ------------------------------------------
        #
        # Bedrock Converse requires the messages list to
        # start with a user message. MAX_HISTORY may otherwise
        # trim away the first user message and leave an assistant
        # message at the beginning.
        #
        # Keep only complete user/assistant pairs plus the
        # current user message.

        if MAX_HISTORY > 0:
            history = history[-MAX_HISTORY:]

        # Safety check: the first message MUST be from the user.
        # If trimming caused an assistant message to become first,
        # remove leading assistant messages.
        while history and history[0].get("role") != "user":
            history.pop(0)


        # ------------------------------------------
        # Call Bedrock
        # ------------------------------------------

        # Final safety check before calling Bedrock.
        if not history or history[0].get("role") != "user":
            raise ValueError("Conversation history must start with a user message.")

        response = bedrock.converse(

            modelId=BEDROCK_MODEL_ID,

            system=[
                {
                    "text": SYSTEM_PROMPT
                }
            ],

            messages=history,

            inferenceConfig={

                "maxTokens": MAX_TOKENS,

                "temperature": TEMPERATURE
            }
        )


        # ------------------------------------------
        # Extract response
        # ------------------------------------------

        answer = (
            response["output"]
            ["message"]
            ["content"][0]
            ["text"]
        )


        # ------------------------------------------
        # Add AI response to history
        # ------------------------------------------

        history.append({

            "role": "assistant",

            "content": [
                {
                    "text": answer
                }
            ]
        })


        # ------------------------------------------
        # Limit history again
        # ------------------------------------------

        history = history[-MAX_HISTORY:]


        # ------------------------------------------
        # Save conversation
        # ------------------------------------------

        session["history"] = history


        logger.info(
            "Bedrock response generated successfully"
        )


        return jsonify({

            "answer": answer,

            "model": BEDROCK_MODEL_ID

        })


    except Exception as e:

        logger.exception(
            "Error while processing Bedrock request"
        )

        return jsonify({

            "error":
            "Unable to generate a response. "
            "Please try again."

        }), 500


# --------------------------------------------------
# Clear chat
# --------------------------------------------------

@app.route("/clear", methods=["POST"])
def clear_chat():

    session.pop(
        "history",
        None
    )

    logger.info(
        "Conversation history cleared"
    )

    return jsonify({
        "message": "Chat cleared"
    })


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )