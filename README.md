---
title: AI Assignment Checker
emoji: 📝
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.24.0"
app_file: app.py
pinned: false
---

# 📝 Code Analyzer

A **Streamlit-based application** to analyze **Python (.py)** and **Jupyter Notebook (.ipynb)** files using **CodeBERT** and **Pylint**.

---

## 📂 Project Structure

- `app.py`: Main Streamlit application.  
- `models/codebert.py`: CodeBERT model loading and analysis logic.  
- `utils/code_utils.py`: Pylint and Jupyter Notebook extraction utilities.  
- `static/uploads/`: Folder for uploaded files.  
- `requirements.txt`: Project dependencies.  

---

## ⚙️ Setup Instructions

### 1. Clone or Set Up the Project
git clone https://github.com/<username>/<repo>.git
cd code_analyzer

### 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:
pip install -r requirements.txt

### 3. Run the Application
streamlit run app.py
Open the provided URL (usually http://localhost:8501) in your browser.

## 🚀 Usage
### 1. Upload a .py or .ipynb file via the Streamlit UI.
### 2. View the analysis report with CodeBERT and Pylint feedback.

## 📝 Notes
### 1. Ensure the static/uploads folder exists (create it manually if needed).
### 2. If using a GPU, CodeBERT will utilize CUDA if available.
### 3.Check logs in the terminal for debugging information.

## 🛠️ Troubleshooting
### 1. Module Not Found → Verify all dependencies are installed:
     pip install -r requirements.txt
### 2. File Upload Issues → Ensure files are valid .py or .ipynb.
### 3.CodeBERT Errors → Check internet connection for model downloading or GPU compatibility.
