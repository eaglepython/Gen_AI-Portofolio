import streamlit as st
import requests
import base64

st.set_page_config(page_title="NLP Text Analysis Demo", layout="centered")
st.title("NLP Text Analysis Demo")

st.markdown("""
Analyze text for sentiment, named entities, and summarization using the NLP API.
""")

api_url = "http://localhost:8000/analyze"

text = st.text_area("Enter text to analyze", "Streamlit is an awesome tool for ML demos!")
task = st.selectbox("Task", ["sentiment", "ner", "summarize"])

if st.button("Analyze"):
    try:
        response = requests.post(api_url, json={"text": text, "task": task})
        if response.status_code == 200:
            result = response.json().get("result", {})
            st.success(f"Result for {task}:")
            st.json(result)
            # Export result button
            import json
            result_str = json.dumps(result, indent=2)
            b64_txt = base64.b64encode(result_str.encode()).decode()
            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="nlp_{task}_result.txt">Download Result</a>'
            st.markdown(href_txt, unsafe_allow_html=True)
        else:
            st.error(f"API error: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.info("Make sure the FastAPI app is running at http://localhost:8000 before using this dashboard.")
