import streamlit as st
import requests
import base64

st.set_page_config(page_title="AI Code Generator Demo", layout="centered")
st.title("AI Code Generator Demo")

st.markdown("""
Generate code from natural language prompts using transformer models.
""")

api_url = "http://localhost:8000/generate"

prompt = st.text_input("Code Prompt", "function to reverse a string")
language = st.text_input("Programming Language", "python")
model = st.selectbox("Model", ["codegen-350M", "codegen-2B", "codebert", "gpt-neo", "gpt2", "starcoder"])

if st.button("Generate Code"):
    try:
        response = requests.post(api_url, json={"prompt": prompt, "language": language, "model": model})
        if response.status_code == 200:
            code = response.json().get("generated_code", "")
            st.success("Generated Code:")
            st.code(code, language=language)
            # Export code button
            b64_txt = base64.b64encode(code.encode()).decode()
            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="generated_code.{language}">Download Code</a>'
            st.markdown(href_txt, unsafe_allow_html=True)
        else:
            st.error(f"API error: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.info("Make sure the FastAPI app is running at http://localhost:8000 before using this dashboard.")
