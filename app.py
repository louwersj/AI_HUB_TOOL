from flask import Flask, request, jsonify
from helperclass import *
import os
import uuid
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def sanitize_json_data(json_data):
    """
    Validate and sanitize JSON data to prevent XSS and other injection attacks.
    """
    try:
        data = json.loads(json_data)
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON structure")
        return data
    except (ValueError, json.JSONDecodeError):
        raise ValueError("Invalid JSON input")

def validate_messages(messages):
    """
    Ensure the messages payload does not contain untrusted or malicious input.
    """
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    for message in messages:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Each message must be a dictionary with 'role' and 'content'")
        if len(message['content']) > 5000:  # Limit content length
            raise ValueError("Message content too long")

@app.route('/report', methods=['POST'])
def report_details():
    try:
        # Handle file uploads
        uploaded_files = request.files.getlist("files")
        pdf_paths = []

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                # Use a secure and unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                pdf_paths.append(file_path)

        # Check if required files are provided
        if not pdf_paths:
            return jsonify({"error": "No valid PDF files uploaded"}), 400

        # Extract JSON data from form-data
        json_data = request.form.get('data')
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate and sanitize JSON input
        data = sanitize_json_data(json_data)
        project_info_payload = data.get('project_info_payload', [])
        cost_info_payload = data.get('cost_info_payload', [])

        documents_content = load_pdf_contents(pdf_paths)

        api_key = os.getenv("OPENAI_API_KEY")
        llm = setup_llm(api_key)

        project_details = extract_project_info(project_info_payload)
        cost_info = extract_cost_info(cost_info_payload)
        question = formulate_question(project_details, cost_info, historical_data=load_historical_data())

        messages = [
            {"role": "system", "content": "You are a QC and architect."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": documents_content}
        ]

        # Validate the messages payload
        validate_messages(messages)

        response = chat_completion(messages, api_key)

        # Remove the uploaded files after processing
        for file_path in pdf_paths:
            os.remove(file_path)

        return response

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
