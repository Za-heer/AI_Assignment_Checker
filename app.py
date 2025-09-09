import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import datetime
from scripts.logger import get_logger
from models.codebert import analyze_with_codebert
from utils.code_utils import pylint_check, extract_code_from_ipynb
from charset_normalizer import detect

logger = get_logger(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.markdown("""
    <style>
    .rainbow {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        font-family: "Comic Sans MS", cursive;
        animation: rainbow 5s infinite;
        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet);
        background-size: 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    @keyframes rainbow {
        0% { background-position: 0%; }
        100% { background-position: 400%; }
    }
    </style>
    <div class="rainbow">Assignment Checker</div>
""", unsafe_allow_html=True)

st.write("Upload your Python (.py) or Jupyter Notebook (.ipynb) file to check its code quality.")

# Input for student ID or name
student_id = st.text_input("Enter Student ID or Name", value="Unknown")

uploaded_files = st.file_uploader("Choose a file", type=['py', 'ipynb'], accept_multiple_files=True)

if uploaded_files is not None:
    results = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        try:
            # Save uploaded file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
        
            st.write(f"File **{uploaded_file.name}** uploaded successfully!")
            is_ipynb = uploaded_file.name.endswith('.ipynb')
        
            with st.spinner("Analyzing code..."):
                # Extract code for .ipynb or read .py
                if is_ipynb:
                    code = extract_code_from_ipynb(file_path)
                    if code is None:
                        st.error("Error: Could not extract code from .ipynb file.")
                        results.append({
                            'Student ID/Name': student_id,
                            'File Name': uploaded_file.name,
                            'CodeBERT Feedback': 'Error: Could not extract code from .ipynb file.',
                            'Pylint Feedback': 'N/A'
                        })
                        continue
                else:
                    # Read file and detect encoding
                    with open(file_path, 'rb') as f:
                        raw_content = f.read()
                    detected = detect(raw_content)
                    encoding = detected['encoding']
               
                    logger.debug(f"Detected encoding for {file_path}: {encoding}")
                    if encoding not in ['utf-8', 'ascii']:
                        code = raw_content.decode(encoding).encode('utf-8').decode('utf-8')
                        # Save as UTF-8 for Pylint
                        temp_file_path = os.path.join(UPLOAD_FOLDER, f"utf8_{uploaded_file.name}")
                        with open(temp_file_path, 'w', encoding='utf-8') as f:
                            f.write(code)
                        file_path = temp_file_path
                        results.append({
                            'Student ID/Name': student_id,
                            'File Name': uploaded_file.name,
                            'CodeBERT Feedback': 'Error: Could not extract code from .ipynb file.',
                            'Pylint Feedback': 'N/A'
                        })
                        continue
                    else:
                        code = raw_content.decode('utf-8')
                
                # For .ipynb files, save extracted code to a temporary .py file
                analysis_file_path = file_path
                if is_ipynb:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                        temp_file.write(code)
                        analysis_file_path = temp_file.name
                        logger.debug(f"Temporary file content:\n{code}")
                
                # Analyze with CodeBERT
                codebert_feedback = analyze_with_codebert(code)
                # Run Pylint
                pylint_feedback = pylint_check(analysis_file_path)
                # Clean up temporary file if created
                if is_ipynb or file_path != os.path.join(UPLOAD_FOLDER, uploaded_file.name):
                    try:
                        os.unlink(analysis_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {analysis_file_path}: {e}")
                results.append({
                            'Student ID/Name': student_id,
                            'File Name': uploaded_file.name,
                            'CodeBERT Feedback': codebert_feedback,
                            'Pylint Feedback': pylint_feedback
                        })
                # Display results
                st.markdown(f"**Analysis Report**\n\n**CodeBERT Feedback:**\n{codebert_feedback}\n\n**Pylint Feedback:**\n```\n{pylint_feedback}\n```")
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            st.error(f"Error processing file: {str(e)}")
            
           
            
    # Save results to Excel
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(UPLOAD_FOLDER, f"analysis_results_{timestamp}.xlsx")
        try:
            df = pd.DataFrame(results)
            df.to_excel(excel_path, index=False)
            st.success(f"Analysis results saved to {excel_path}")
            # Provide download link
            with open(excel_path, 'rb') as f:
                st.download_button(
                    label="Download Analysis Results",
                    data=f,
                    file_name=f"analysis_results_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            logger.error(f"Error saving results to Excel: {e}")
            st.error(f"Error saving results to Excel: {str(e)}")