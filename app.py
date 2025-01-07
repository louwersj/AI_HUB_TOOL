from flask import Flask, request, jsonify
from helperclass import DataLoader, chat_completion, extract_project_info, extract_cost_info, formulate_question, count_tokens
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

        json_data = request.form.get('data')
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        data = json.loads(json_data)
        project_info_payload = data.get('project_info_payload', [])
        cost_info_payload = data.get('cost_info_payload', [])

        api_key = os.getenv("OPENAI_API_KEY")
        documents_content = DataLoader.load_pdf_contents(pdf_paths, api_key)

        project_details = extract_project_info(project_info_payload)
        cost_info = extract_cost_info(cost_info_payload)
        historical_data = DataLoader.load_historical_data()
        question = formulate_question(project_details, cost_info, historical_data=historical_data)

        system_msg = "You are a QC and architect."
        system_tokens = count_tokens(system_msg)
        user_tokens = count_tokens(question)
        assistant_tokens = count_tokens(documents_content)
        total_tokens = system_tokens + user_tokens + assistant_tokens

        print(f"Total tokens: {total_tokens}")

        MAX_TOKENS = 7000
        if total_tokens > MAX_TOKENS:
            ratio = (MAX_TOKENS - system_tokens) / total_tokens
            user_tokens_allowed = int(user_tokens * ratio)
            assistant_tokens_allowed = int(assistant_tokens * ratio)
            encoding = tiktoken.encoding_for_model('gpt-4')
            question = encoding.decode(encoding.encode(question)[:user_tokens_allowed])
            documents_content = encoding.decode(encoding.encode(documents_content)[:assistant_tokens_allowed])

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": documents_content}
        ]

        response = chat_completion(messages, api_key)

        for file_path in pdf_paths:
            os.remove(file_path)

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
