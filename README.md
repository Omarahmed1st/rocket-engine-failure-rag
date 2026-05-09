# 🚀 Rocket Engine Failure Analysis RAG System

A semantic Retrieval-Augmented Generation (RAG) system for analyzing rocket engine thruster test reports using NLP, FAISS vector search, and Streamlit.

---

## 📌 Overview

This project enables semantic search over rocket engine telemetry and thruster firing test reports.

Instead of relying on exact keyword matching, the system understands the meaning of user queries using sentence embeddings and retrieves the most relevant engineering reports.

Example queries:

* `Show tests with anomaly 22`
* `High thrust tests`
* `Low thrust behavior`
* `Stable valve activity`
* `Engine instability`

---

## 🧠 Features

✅ Semantic Search using Sentence Transformers
✅ Fast Vector Retrieval with FAISS
✅ Rocket Engine Failure Analysis
✅ Anomaly Detection Retrieval
✅ Interactive Streamlit Dashboard
✅ Real Engineering Telemetry Dataset
✅ Metadata-Aware Search Results
✅ Natural Language Query Support

---

## 🛠 Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Transformers
* Pandas
* NumPy
* Scikit-learn

---

## 🏗 System Architecture

User Query
↓
Sentence Transformer Embedding
↓
FAISS Vector Similarity Search
↓
Top-k Relevant Test Reports
↓
Streamlit UI Display

---

## 📂 Project Structure

```bash
rocket-engine-failure-rag/
│
├── app.py
├── build_index.py
├── requirements.txt
├── packages.txt
├── .gitignore
│
├── models/
│   ├── faiss_index.bin
│   └── documents.pkl
│
├── utils/
│   └── summarizer.py
│
└── data/
    └── dataset files
```

---

## 🚀 Live Demo

🔗 Streamlit App:

https://rocket-engine-failure-rag-5jflwj82ambecvcoqjwkc3.streamlit.app/

---

## 💻 GitHub Repository

🔗 GitHub:

https://github.com/Omarahmed1st/rocket-engine-failure-rag

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Omarahmed1st/rocket-engine-failure-rag.git
cd rocket-engine-failure-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Example Results

The system can successfully retrieve reports related to:

* unstable valve activity
* propulsion instability
* high thrust behavior
* low thrust behavior
* anomaly-based failures

using semantic similarity instead of exact keyword search.

---

## 🎯 Learning Outcomes

This project helped me understand:

* NLP embeddings
* Vector databases
* Information Retrieval systems
* Semantic search
* Retrieval-Augmented Generation (RAG)
* FAISS indexing
* Engineering telemetry analysis

---

## 📸 Screenshots

*Add screenshots here after deployment.*

---

## 👨‍💻 Author

Omar Ahmed

LinkedIn:
https://www.linkedin.com/

GitHub:
https://github.com/Omarahmed1st

---

## ⭐ Future Improvements

* Add telemetry charts and visualizations
* Integrate LLM-generated explanations
* Add filtering by anomaly type
* Add downloadable engineering reports
* Support conversational memory

---

## 📜 License

This project is open-source and available under the MIT License.
