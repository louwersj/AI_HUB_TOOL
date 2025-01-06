from flask import Flask, request, jsonify
from helperclass import DataLoader, chat_completion, extract_project_info, extract_cost_info, formulate_question
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/report', methods=['POST'])
def report_details():
    try:
        # Handle uploaded files
        uploaded_files = request.files.getlist("files")
        pdf_paths = []

        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                pdf_paths.append(file_path)

        if not pdf_paths:
            return jsonify({"error": "No valid PDF files uploaded"}), 400

        # Handle JSON data
        json_data = request.form.get('data')
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        data = json.loads(json_data)

        # Extract payloads
        project_info_payload = data.get('project_info_payload', [])
        cost_info_payload = data.get('cost_info_payload', [])

        # Validate payloads
        if not isinstance(project_info_payload, list) or not isinstance(cost_info_payload, list):
            return jsonify({"error": "Invalid payload structure"}), 400

        # Process data
        project_details = extract_project_info(project_info_payload)
        cost_info = extract_cost_info(cost_info_payload)

        # Load documents content
        documents_content = DataLoader.load_pdf_contents(pdf_paths)

        # Prepare OpenAI API request
        api_key = os.getenv("OPENAI_API_KEY")
        historical_data = DataLoader.load_historical_data()

        # Split data into chunks
        chunks = []
        chunk_size = 2000  # Approximate token limit per chunk
        all_data = [
            {"type": "historical_data", "content": historical_data},
            {"type": "documents_content", "content": documents_content},
            {"type": "cost_info", "content": cost_info},
            {"type": "project_details", "content": project_details},
        ]

        for data_entry in all_data:
            content = str(data_entry["content"])
            for i in range(0, len(content), chunk_size):
                chunks.append({"type": data_entry["type"], "content": content[i:i + chunk_size]})

        # Process each chunk and collect responses
        aggregated_responses = {}
        for i, chunk in enumerate(chunks):
            question = f"Process the following {chunk['type']} chunk:\n{chunk['content']}"
            messages = [
                {"role": "system", "content": "You are an expert architect and cost estimator."},
                {"role": "user", "content": question}
            ]
            response = chat_completion(messages, api_key)
            if chunk["type"] not in aggregated_responses:
                aggregated_responses[chunk["type"]] = []
            aggregated_responses[chunk["type"]].append(response)

        # Clean up uploaded files
        for file_path in pdf_paths:
            os.remove(file_path)

        return jsonify({"response": aggregated_responses})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
