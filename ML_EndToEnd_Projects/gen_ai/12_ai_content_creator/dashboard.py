import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import io
import base64

st.set_page_config(page_title="AI Content Creator Demo", layout="centered")
st.title("AI Content Creator Demo")

st.markdown("""
Generate text or images from prompts using state-of-the-art models.
""")

api_url_text = "http://localhost:8000/generate/text"
api_url_image = "http://localhost:8000/generate/image"

st.warning(
    "This dashboard requires the FastAPI backend to be running.\n\n"
    "**Start it with:**\n"
    "    uvicorn app:app --reload\n"
    "from this folder in a separate terminal window."
)


# Sidebar for extra controls and visuals
st.sidebar.header("Visualization & Controls")
st.sidebar.markdown("---")
st.sidebar.write("Adjust generation parameters and view analytics.")

mode = st.radio("Mode", ["Text", "Image"])

if mode == "Text":
    prompt = st.text_area("Text Prompt", "Write a poem about AI", height=100)
    model = st.selectbox("Model", ["gpt2", "gpt-neo", "gpt-j", "flan-t5", "llama2"])
    max_length = st.sidebar.slider("Max Length", 32, 512, 128, step=8)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
    top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.95, step=0.01)
    show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis", value=True)
    show_word_count = st.sidebar.checkbox("Show Word Count", value=True)
    show_length_chart = st.sidebar.checkbox("Show Length Chart", value=True)
    if st.button("Generate Text"):
        try:
            response = requests.post(api_url_text, json={
                "prompt": prompt,
                "model": model,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            })
            if response.status_code == 200:
                text = response.json().get("generated_text", "")
                st.success("Generated Text:")
                st.write(text)
                # Visuals
                if show_word_count:
                    wc = len(text.split())
                    st.info(f"Word count: {wc}")
                if show_sentiment:
                    blob = TextBlob(text)
                    st.write(f"Sentiment: Polarity {blob.sentiment.polarity:.2f}, Subjectivity {blob.sentiment.subjectivity:.2f}")
                if show_length_chart:
                    lengths = [len(w) for w in text.split()]
                    fig, ax = plt.subplots()
                    ax.hist(lengths, bins=range(1, max(lengths)+2), color='skyblue', edgecolor='black')
                    ax.set_title('Word Length Distribution')
                    ax.set_xlabel('Word Length')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                    # Export chart button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="word_length_chart.png">Download Chart as PNG</a>'
                    st.markdown(href, unsafe_allow_html=True)
                # Export text button
                b64_txt = base64.b64encode(text.encode()).decode()
                href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="generated_text.txt">Download Text Result</a>'
                st.markdown(href_txt, unsafe_allow_html=True)
            else:
                st.error(f"API error: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    prompt = st.text_input("Image Prompt", "A futuristic cityscape")
    model = st.selectbox("Model", ["stable-diffusion", "sdxl"])
    num_images = st.sidebar.slider("Number of Images", 1, 4, 1)
    width = st.sidebar.slider("Width", 256, 1024, 512, step=64)
    height = st.sidebar.slider("Height", 256, 1024, 512, step=64)
    show_grid = st.sidebar.checkbox("Show as Grid", value=True)
    if st.button("Generate Image"):
        try:
            response = requests.post(api_url_image, json={
                "prompt": prompt,
                "model": model,
                "num_images": num_images,
                "width": width,
                "height": height
            })
            if response.status_code == 200:
                images = response.json().get("images", [])
                if show_grid and len(images) > 1:
                    cols = st.columns(len(images))
                    for i, img_b64 in enumerate(images):
                        with cols[i]:
                            st.image(img_b64)
                            # Export image button
                            img_data = img_b64.split(",", 1)[-1]
                            href_img = f'<a href="data:image/png;base64,{img_data}" download="generated_image_{i+1}.png">Download Image {i+1}</a>'
                            st.markdown(href_img, unsafe_allow_html=True)
                else:
                    for idx, img_b64 in enumerate(images):
                        st.image(img_b64)
                        img_data = img_b64.split(",", 1)[-1]
                        href_img = f'<a href="data:image/png;base64,{img_data}" download="generated_image_{idx+1}.png">Download Image {idx+1}</a>'
                        st.markdown(href_img, unsafe_allow_html=True)
            else:
                st.error(f"API error: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

st.info("Make sure the FastAPI app is running at http://localhost:8000 before using this dashboard.")
