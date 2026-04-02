import streamlit as st
import pickle


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    h1, h2, h3 {
        color: #1a1a1a !important;
        font-family: "Segoe UI", sans-serif;
        font-weight: 700;
    }
    .stTextArea > label {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    .stSelectbox > label {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a6fd8 0%, #6a4190 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        font-weight: 700;
        font-size: 1.3em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .spam {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #f44336;
        color: #b71c1c !important;
    }
    .ham {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 6px solid #4caf50;
        color: #1b5e20 !important;
    }
    .confidence-text {
        color: #2c3e50 !important;
        font-size: 1.1em;
        font-weight: 600;
    }
    
    /* Sidebar white text fix */
    .css-1d391kg [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    .css-1d391kg [data-testid="stSidebar"] h1, 
    .css-1d391kg [data-testid="stSidebar"] h2, 
    .css-1d391kg [data-testid="stSidebar"] h3,
    .css-1d391kg [data-testid="stSidebar"] p,
    .css-1d391kg [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    .css-1d391kg [data-testid="stSidebar"] strong {
        color: #e0e0e0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_vectorizer():
    """Load all models and TF‑IDF vectorizer from pickle files."""
    models = {
        "MultinomialNB": pickle.load(open("models/naive_bayes_model.pkl", "rb")),
        "SVM": pickle.load(open("models/svm_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("models/logistic_regression_model.pkl", "rb")),
    }
    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    return models, vectorizer


models, vectorizer = load_model_and_vectorizer()


# ----- UI SETUP -----

st.title("📧 Spam Email Detection using NLP")
st.markdown(
    "<div style='font-size:1.15em; color:#2c3e50; margin-bottom:25px; line-height:1.6;'>"
    "Enter the email text and select the model to classify it. "
    "We'll show whether it's <b>Spam</b> or <b>Ham</b> along with confidence score."
    "</div>",
    unsafe_allow_html=True,
)

# Input section
st.subheader("Input Email Text")
input_text = st.text_area("Paste your email or SMS text here...", height=160, 
                         placeholder="Subject: Free Money Alert!\n\nDear Customer,\nYou've won $1,000,000...")

# Model selector
st.subheader("Choose Model")
selected_model_name = st.selectbox(
    "Select classification model:",
    options=["MultinomialNB", "SVM (Best Estimate)", "Logistic Regression"],
    help="Choose the model used for spam detection.",
)

# Predict button
if st.button("Predict Spam / Ham", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        try:
            # Vectorize input
            X = vectorizer.transform([input_text])

            # Get model
            model = models[selected_model_name]

            # Predict class (0 = ham, 1 = spam)
            pred = model.predict(X)[0]

            # Show result box
            if pred == 1:
                st.markdown(
                    """
                    <div class="result-box spam">
                        <b>SPAM DETECTED!</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-box ham">
                        <b>HAM (Not Spam)</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show confidence if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
                st.progress(confidence, text="Confidence level")
                st.markdown(
                    f"<div class='confidence-text'>"
                    f"Model Confidence: <b>{confidence:.3f}</b> "
                    f"(higher value = more certain about prediction)</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Model does not support probability (no confidence shown).")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.code(str(e))

# Optional: model info in sidebar
st.sidebar.header("Model Information")
st.sidebar.markdown(
    """
    **Models Used:**
    - MultinomialNB  
    - SVM  
    - Logistic Regression  

    **Features:** TF‑IDF Vectorization  
    **Labels:** `0 = Ham`, `1 = Spam`
    """,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Built with Streamlit | Spam Detection Project</small>",
    unsafe_allow_html=True,
)