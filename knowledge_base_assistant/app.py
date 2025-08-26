from flask import Flask, request, jsonify, render_template, g, make_response, redirect, url_for
from flask_babel import Babel
from flasgger import Swagger
import uuid
import os

from knowledge_base_assistant import db
from knowledge_base_assistant import rag


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize Swagger for API docs
# swagger = Swagger(app)

# Setup Babel for i18n
babel = Babel(app)

ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', '1') == '1'


# User's language
def get_locale():
    # 1. Language cookie
    lang = request.cookies.get('lang')
    if lang in ['en', 'es', 'it']:
        return lang
    # 2. Browser's header
    return request.accept_languages.best_match(['en', 'es', 'it'])


babel = Babel(app, locale_selector=get_locale)


@app.before_request
def before_request():
    g.locale = str(get_locale())


@app.route('/set_language/<lang>')
def set_language(lang):
    if lang not in ['en', 'es', 'it']:
        return redirect(url_for('home'))

    response = make_response(redirect(request.referrer or url_for('home')))
    response.set_cookie('lang', lang, max_age=30 * 24 * 60 * 60)
    return response


@app.route('/')
def home():
    """Renders the main question-answering web page."""
    grafana_base_url = os.getenv("GRAFANA_BASE_URL")
    grafana_dashboard_uid = os.getenv("GRAFANA_DASHBOARD_UID")
    
    return render_template(
        'index.html', 
        grafana_base_url=grafana_base_url,
        grafana_dashboard_uid=grafana_dashboard_uid
    )


# API Endpoints

# 1. Ask question
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
            enable_monitoring:
              type: boolean
              example: false
              description: "Optional: Enable detailed monitoring for this request"
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
            metrics:
              type: object
              description: "Optional: Detailed metrics (only if monitoring enabled)"
              properties:
                processing_time_seconds:
                  type: number
                  example: 2.34
                total_cost_usd:
                  type: number
                  example: 0.002156
                total_tokens:
                  type: object
                  properties:
                    prompt_tokens:
                      type: integer
                      example: 150
                    completion_tokens:
                      type: integer
                      example: 75
                    total_tokens:
                      type: integer
                      example: 225
                relevance_evaluation:
                  type: object
                  properties:
                    Relevance:
                      type: string
                      example: "RELEVANT"
                    Explanation:
                      type: string
                      example: "The answer directly addresses the user's question"
                model_used:
                  type: string
                  example: "gpt-4o-mini"
                search_results_count:
                  type: integer
                  example: 5
    """
    data = request.get_json()
    question = data.get("question")
    enable_request_monitoring = data.get("enable_monitoring", False)

    user_language = g.get('locale', 'en')

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    use_monitoring = ENABLE_MONITORING or enable_request_monitoring

    try:
        if use_monitoring:
            answer, context, metrics = rag.answer_question(
                question, 
                user_language=user_language, 
                evaluate=True
            )
        else:
            answer, context = rag.answer_question(question, user_language=user_language)
            metrics = None

        conversation_id = str(uuid.uuid4())

        db.save_conversation(
            conversation_id, 
            question, 
            answer, 
            context, 
            metrics=metrics, 
            user_language=user_language
        )

        response_data = {
            "question": question,
            "answer": answer,
            "context": context,
            "conversation_id": conversation_id
        }

        if use_monitoring and metrics:
            response_data["metrics"] = metrics

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "An error occurred while processing your question"}), 500


# 2. Submit feedback
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

    try:
        db.save_feedback(conversation_id, feedback)

        return jsonify({
            "conversation_id": conversation_id,
            "feedback": feedback,
            "status": "feedback received"
        })
    except Exception as e:
        app.logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({"error": "An error occurred while saving feedback"}), 500


# 3. Conversation history
@app.route('/conversations', methods=['GET'])
def get_conversations():
    """
    Get recent conversations with optional filtering
    ---
    tags:
      - Monitoring
    parameters:
      - name: limit
        in: query
        type: integer
        default: 10
        description: Maximum number of conversations to return
      - name: relevance
        in: query
        type: string
        enum: [RELEVANT, PARTLY_RELEVANT, NON_RELEVANT]
        description: Filter by relevance level
      - name: language
        in: query
        type: string
        enum: [en, es, it]
        description: Filter by user language
    responses:
      200:
        description: List of recent conversations
        schema:
          type: object
          properties:
            conversations:
              type: array
              items:
                type: object
            count:
              type: integer
    """
    limit = min(int(request.args.get('limit', 10)), 100)
    relevance = request.args.get('relevance')
    language = request.args.get('language')

    try:
        conversations = db.get_recent_conversations(
            limit=limit,
            relevance=relevance,
            user_language=language
        )
        
        conversations_list = [dict(conv) for conv in conversations]
        
        return jsonify({
            "conversations": conversations_list,
            "count": len(conversations_list)
        })
    except Exception as e:
        app.logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({"error": "An error occurred while fetching conversations"}), 500


# 4. Feedback stats
@app.route('/stats/feedback', methods=['GET'])
def get_feedback_statistics():
    """
    Get feedback statistics
    ---
    tags:
      - Monitoring
    responses:
      200:
        description: Feedback statistics
        schema:
          type: object
          properties:
            thumbs_up:
              type: integer
              example: 25
            thumbs_down:
              type: integer
              example: 3
            total_feedback:
              type: integer
              example: 28
            positive_ratio:
              type: number
              example: 0.89
    """
    try:
        stats = db.get_feedback_stats()
        total = stats['thumbs_up'] + stats['thumbs_down']
        positive_ratio = stats['thumbs_up'] / total if total > 0 else 0

        return jsonify({
            "thumbs_up": stats['thumbs_up'],
            "thumbs_down": stats['thumbs_down'],
            "total_feedback": total,
            "positive_ratio": round(positive_ratio, 2)
        })
    except Exception as e:
        app.logger.error(f"Error fetching feedback stats: {str(e)}")
        return jsonify({"error": "An error occurred while fetching statistics"}), 500


# 5. Conversation analytics
@app.route('/stats/conversations', methods=['GET'])
def get_conversation_statistics():
    """
    Get conversation analytics
    ---
    tags:
      - Monitoring
    parameters:
      - name: days
        in: query
        type: integer
        default: 7
        description: Number of days to include in statistics
    responses:
      200:
        description: Conversation analytics
        schema:
          type: object
          properties:
            total_conversations:
              type: integer
              example: 150
            avg_response_time:
              type: number
              example: 2.34
            total_cost:
              type: number
              example: 0.125
            avg_cost_per_conversation:
              type: number
              example: 0.00083
            total_tokens_used:
              type: integer
              example: 45000
            relevance_distribution:
              type: object
              properties:
                relevant:
                  type: integer
                  example: 120
                partly_relevant:
                  type: integer
                  example: 25
                non_relevant:
                  type: integer
                  example: 5
            language_distribution:
              type: array
              items:
                type: object
                properties:
                  user_language:
                    type: string
                    example: "en"
                  count:
                    type: integer
                    example: 100
    """
    days = int(request.args.get('days', 7))
    
    try:
        stats = db.get_conversation_stats(days=days)
        
        return jsonify({
            "period_days": days,
            **stats
        })
    except Exception as e:
        app.logger.error(f"Error fetching conversation stats: {str(e)}")
        return jsonify({"error": "An error occurred while fetching analytics"}), 500


# 6. Specific conversation
@app.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """
    Get a specific conversation by ID
    ---
    tags:
      - Monitoring
    parameters:
      - name: conversation_id
        in: path
        type: string
        required: true
        description: The conversation ID
    responses:
      200:
        description: Conversation details
        schema:
          type: object
      404:
        description: Conversation not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Conversation not found"
    """
    try:
        conversation = db.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify(dict(conversation))
    except Exception as e:
        app.logger.error(f"Error fetching conversation {conversation_id}: {str(e)}")
        return jsonify({"error": "An error occurred while fetching the conversation"}), 500


# Health check
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            monitoring_enabled:
              type: boolean
              example: true
    """
    try:
        db.get_db_connection().close()
        
        return jsonify({
            "status": "healthy",
            "monitoring_enabled": ENABLE_MONITORING
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)