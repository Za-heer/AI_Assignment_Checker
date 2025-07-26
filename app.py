from flask import Flask, request, render_template
import os
from transformers import AutoTokenizer, AutoModel
import torch
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CodeBERT globally (once at startup)
tokenizer = None
model = None
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    if torch.cuda.is_available():
        model.to('cuda')
    print("CodeBERT loaded successfully.")
except Exception as e:
    print(f"Error loading CodeBERT: {e}")

def pylint_check(file_path):
    try:
        output = StringIO()
        reporter = TextReporter(output)
        Run([file_path, '--disable=C0114,C0115,C0116,R0903,W0311,W0703,C0303,C0301', '--max-line-length=120'], reporter=reporter, exit=False)
        pylint_output = output.getvalue()
        return pylint_output if pylint_output.strip() else "No critical issues found. Code looks good!"
    except Exception as e:
        return f"Pylint error: {str(e)}"

def analyze_code(file_path):
    try:
        # Read the code file
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        
        # CodeBERT analysis
        codebert_feedback = "CodeBERT not loaded."
        if tokenizer and model:
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            codebert_feedback = f"Code analyzed with CodeBERT. Length: {len(code)} characters."

        # Pylint analysis
        pylint_feedback = pylint_check(file_path)
        
        # Combine and format feedback
        feedback = f"<h3>Analysis Report</h3><p><strong>CodeBERT Feedback:</strong> {codebert_feedback}</p><p><strong>Pylint Feedback:</strong><br><pre>{pylint_feedback}</pre></p>"
        return feedback
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        if file and (file.filename.endswith('.py') or file.filename.endswith('.ipynb')):
            file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            file.save(file_path)
            feedback = analyze_code(file_path)
            return f'<h2>File {file.filename} uploaded successfully!</h2>{feedback}'
        return 'Invalid file type', 400
    except Exception as e:
        return f'Error during upload: {str(e)}', 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5001)