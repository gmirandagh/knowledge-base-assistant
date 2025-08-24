from flask import Flask, request, jsonify, render_template, g, make_response, redirect, url_for
from flask_babel import Babel
from flasgger import Swagger
import uuid
import rag  # rag.py

app = Flask(__name__)
# swagger = Swagger(app)
babel = Babel(app)

# @app.route('/')
# def home():
#     """Redirect to the API documentation."""
#     return redirect('/apidocs')

# User's language
def get_locale():
    # 1. Check for language cookie
    lang = request.cookies.get('lang')
    if lang in ['en', 'es', 'it']:
        return lang
    # 2. If no cookie, fall back to the browser's header
    return request.accept_languages.best_match(['en', 'es', 'it'])

babel = Babel(app, locale_selector=get_locale)

# Run before each request to set language
@app.before_request
def before_request():
    g.locale = str(get_locale())

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang not in ['en', 'es', 'it']:
        return redirect(url_for('home'))

    response = make_response(redirect(request.referrer or url_for('home')))
    response.set_cookie('lang', lang, max_age=30*24*60*60)
    return response

# Web page from the root URL
@app.route('/')
def home():
    """Renders the main question-answering web page."""
    return render_template('index.html')


# --- API Endpoints ---

# --- 1. Ask a question ---
@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Ask a question and get an answer along with retrieved sources
    ---
    tags:
      - Q&A
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: "Who are the authors of Document X?"
    responses:
      200:
        description: Answer with conversation ID and context
        schema:
          type: object
          properties:
            conversation_id:
              type: string
              example: "123e4567-e89b-12d3-a456-426614174000"
            answer:
              type: string
              example: "The authors of Document X are Alice Smith and Bob Johnson."
            context:
              type: array
              items:
                type: object
                properties:
                  type:
                    type: string
                    example: "metadata"
                  title:
                    type: string
                    example: "Document X"
                  section_title:
                    type: string
                    example: "Introduction"
                  page_number:
                    type: string
                    example: "1"
                  content:
                    type: string
                    example: "This is a content chunk from Document X."
                  authors:
                    type: array
                    items:
                      type: string
                    example: ["Alice Smith", "Bob Johnson"]
                  year:
                    type: string
                    example: "2023"
    """
    data = request.get_json()
    question = data.get("question")
    
    # Get user's language
    user_language = g.get('locale', 'en')

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Pass detected language to RAG
    answer, context = rag.answer_question(question, user_language=user_language)
    
    conversation_id = str(uuid.uuid4())

    return jsonify({
        "question": question,
        "answer": answer,
        "context": context,
        "conversation_id": conversation_id
    })


# --- 2. Submit feedback ---
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback for a conversation
    ---
    tags:
      - Feedback
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            conversation_id:
              type: string
              example: "123e4567-e89b-12d3-a456-426614174000"
            feedback:
              type: integer
              enum: [-1, 1]
              example: 1
    responses:
      200:
        description: Feedback acknowledged
        schema:
          type: object
          properties:
            status:
              type: string
              example: "feedback received"
            conversation_id:
              type: string
              example: "123e4567-e89b-12d3-a456-426614174000"
            feedback:
              type: integer
              example: 1
    """
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    feedback = data.get("feedback")

    if not conversation_id or feedback not in [-1, 1]:
        return jsonify({"error": "Missing or invalid parameters"}), 400

    return jsonify({
        "conversation_id": conversation_id,
        "feedback": feedback,
        "status": "feedback received"
    })


if __name__ == '__main__':
    app.run(debug=True)




# How to test
# Install dependencies if you haven’t already:
# pip install flask flasgger

# Run the Flask app:
# python app.py

# Open Swagger UI in your browser:
# http://127.0.0.1:5000/apidocs

# You’ll see interactive forms for:
# /ask → enter a question, get answer + conversation_id.
# /feedback → enter conversation_id and feedback (+1 or -1).