import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Rocket Engine Failure Analysis",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp{
        background-color:#020817;
        color:white;
    }

    .main-title{
        font-size:48px;
        font-weight:800;
        color:white;
        margin-bottom:10px;
    }

    .subtitle{
        font-size:18px;
        color:#94a3b8;
        margin-bottom:40px;
    }

    .result-title{
        font-size:30px;
        font-weight:700;
        color:white;
        margin-top:30px;
        margin-bottom:15px;
    }

    .result-card{
        background:#052e16;
        border:1px solid #22c55e;
        border-radius:18px;
        padding:25px;
        margin-bottom:25px;
        box-shadow:0px 0px 15px rgba(34,197,94,0.2);
    }

    .result-text{
        color:white;
        font-size:17px;
        line-height:1.9;
    }

    .stTextInput input{
        background-color:#0f172a;
        color:white;
        border:2px solid #334155;
        border-radius:12px;
        padding:12px;
        font-size:18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
# =====================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# LOAD FAISS INDEX
# =====================================================

index = faiss.read_index(
    "models/faiss_index.bin"
)

# =====================================================
# LOAD DOCUMENTS
# =====================================================

with open(
    "models/documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

# =====================================================
# SEARCH FUNCTION
# =====================================================

def search(query, top_k=5):

    query_lower = query.lower()

    # =================================================
    # SPECIAL CASE:
    # anomaly search
    # =================================================

    if "anomaly" in query_lower:

        words = query_lower.split()

        anomaly_number = None

        for word in words:

            if word.isdigit():

                anomaly_number = word

        if anomaly_number is not None:

            filtered_results = []

            for doc in documents:

                if f"Detected anomalies: [{anomaly_number}]" in doc:

                    filtered_results.append(doc)

            if len(filtered_results) > 0:

                return filtered_results[:top_k]

    # =================================================
    # NORMAL SEMANTIC SEARCH
    # =================================================

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(documents[idx])

    return results

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="main-title">
        🚀 Rocket Engine Failure Analysis RAG System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Semantic Search over Rocket Thruster Test Reports
        using FAISS + Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# EXAMPLE QUESTIONS
# =====================================================

st.markdown(
    """
    ### Example Questions

    - Show tests with anomaly 22
    - High thrust tests
    - Low thrust behavior
    - Stable valve activity
    - Unstable valve activity
    - Propulsion behavior
    """
)

# =====================================================
# USER INPUT
# =====================================================

query = st.text_input(
    "Ask your question:"
)

# =====================================================
# RUN SEARCH
# =====================================================

if query:

    results = search(query)

    st.markdown("## 🔎 Top Results")

    if len(results) == 0:

        st.warning(
            "No matching results found."
        )

    else:

        for i, result in enumerate(results):

            st.markdown(
                f"""
                <div class="result-title">
                    Result {i+1}
                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================================
            # CLEAN TEXT
            # =========================================

            lines = result.split("\n")

            clean_lines = []

            for line in lines:

                line = line.strip()

                if line != "":

                    clean_lines.append(line)

            formatted_result = "<br>".join(
                clean_lines
            )

            # =========================================
            # SHOW RESULT CARD
            # =========================================

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-text">
                        {formatted_result}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )