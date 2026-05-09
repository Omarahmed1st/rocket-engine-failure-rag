import os
import pickle
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# PATHS
# -----------------------------
DATA_FOLDER = "data/dataset/dataset/train"

documents = []

# -----------------------------
# BUILD DOCUMENTS
# -----------------------------
for file in os.listdir(DATA_FOLDER):

    # skip non csv
    if not file.endswith(".csv"):
        continue

    # skip metadata
    if file == "metadata.csv":
        continue

    file_path = os.path.join(DATA_FOLDER, file)

    try:

        df = pd.read_csv(file_path)

        # basic statistics
        avg_thrust = df["thrust"].mean()
        max_thrust = df["thrust"].max()

        avg_mfr = df["mfr"].mean()

        avg_vl = df["vl"].mean()

        anomalies = (
            df["anomaly_code"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        # -----------------------------
        # THRUST DESCRIPTION
        # -----------------------------
        if avg_thrust > 1:
            thrust_desc = "high thrust behavior"

        elif avg_thrust > 0.5:
            thrust_desc = "moderate thrust behavior"

        else:
            thrust_desc = "low thrust behavior"

        # -----------------------------
        # VALVE DESCRIPTION
        # -----------------------------
        if avg_vl > 0.5:
            valve_desc = "unstable valve activity"

        else:
            valve_desc = "stable valve activity"

        # -----------------------------
        # ANOMALY DESCRIPTION
        # -----------------------------
        if len(anomalies) > 0:
            anomaly_text = f"Detected anomalies: {anomalies}"

        else:
            anomaly_text = "No anomalies were detected"

        # -----------------------------
        # FINAL DOCUMENT
        # -----------------------------
        text = f"""
Thruster firing test analysis.

File name: {file}

The test demonstrated {thrust_desc}.

Average thrust level was {avg_thrust:.4f}.

Maximum thrust reached {max_thrust:.4f}.

Average mass flow rate was {avg_mfr:.4f}.

The propulsion system showed {valve_desc}.

Average valve level was {avg_vl:.4f}.

{anomaly_text}.

The thruster firing sequence showed propulsion activity, thrust variation, and valve response changes.
"""

        documents.append(text)

    except Exception as e:
        print(f"Error in {file}: {e}")

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -----------------------------
# SAVE INDEX
# -----------------------------
os.makedirs("models", exist_ok=True)

faiss.write_index(
    index,
    "models/faiss_index.bin"
)

# -----------------------------
# SAVE DOCUMENTS
# -----------------------------
with open("models/documents.pkl", "wb") as f:

    pickle.dump(documents, f)

print("\nFAISS index saved successfully!")

print(f"Indexed documents: {len(documents)}")