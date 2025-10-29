import streamlit as st
import requests
import base64

st.set_page_config(page_title="E-commerce Recommender Demo", layout="centered")
st.title("E-commerce Recommender System Demo")

st.markdown("""
Interact with the recommender system. Enter a user ID to get personalized product recommendations.
""")

api_url = "http://localhost:8000/recommend"

user_id = st.text_input("User ID", "1")

if st.button("Get Recommendations"):
    try:
        response = requests.post(api_url, json={"user_id": user_id})
        if response.status_code == 200:
            recs = response.json().get("recommendations", [])
            st.success(f"Recommendations for user {user_id}:")
            for rec in recs:
                st.write(f"- {rec}")
            # Export recommendations button
            recs_text = '\n'.join(recs)
            b64_txt = base64.b64encode(recs_text.encode()).decode()
            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="recommendations_user_{user_id}.txt">Download Recommendations</a>'
            st.markdown(href_txt, unsafe_allow_html=True)
        else:
            st.error(f"API error: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.info("Make sure the FastAPI app is running at http://localhost:8000 before using this dashboard.")
