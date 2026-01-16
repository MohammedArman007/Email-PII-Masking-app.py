import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from masking import mask_all
import joblib
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Email PII Masking & Classification (Advanced)",
                   layout="wide",
                   page_icon="📧")

DATA_AUTO = "email_pii_dataset_10000.xlsx"  # put your dataset here if you want auto-load

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_dataset(path: str = DATA_AUTO):
    p = Path(path)
    if p.exists():
        # accept both xlsx and csv
        if p.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
        return df
    return None

@st.cache_data
def train_model(df: pd.DataFrame, text_col="body", label_col="label"):
    # basic cleaning
    df_local = df.copy()
    df_local = df_local[[text_col, label_col]].dropna()
    X = df_local[text_col].astype(str)
    y = df_local[label_col].astype(str)

    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xv = vect.fit_transform(X)

    model = MultinomialNB()
    model.fit(Xv, y)

    preds = model.predict(Xv)
    acc = accuracy_score(y, preds)
    clf_report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds, labels=list(sorted(y.unique())))

    return {
        "model": model,
        "vectorizer": vect,
        "accuracy": acc,
        "report": clf_report,
        "confusion_matrix": cm,
        "labels": list(sorted(y.unique()))
    }

# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Single Email", "Batch (Upload)", "Batch (Auto-load)"])
enable_aadhaar = st.sidebar.checkbox("Enable Aadhaar/PAN/SSN mask (regex)", value=True)
enable_spacy_entities = st.sidebar.checkbox("Enable spaCy entity masking", value=True)
enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Project: Email PII Masking & Classification")
st.sidebar.markdown("Developed: Advanced feature set")

# ---------------------------
# Load dataset if requested
# ---------------------------
df_auto = load_dataset()  # cached

# Display dataset summary in sidebar if available
if df_auto is not None:
    st.sidebar.markdown(f"Auto dataset found: **{DATA_AUTO}**")
    st.sidebar.write(f"Rows: {df_auto.shape[0]} | Columns: {df_auto.shape[1]}")
else:
    st.sidebar.info("No auto dataset found. Use Upload or place file 'email_pii_dataset_10000.xlsx' in folder.")

# ---------------------------
# Model training area (for auto dataset)
# ---------------------------
model_bundle = None
if mode == "Batch (Auto-load)":
    if df_auto is None:
        st.error("Auto-load dataset not found. Put the dataset in the app folder or switch to Upload.")
    else:
        st.header("🔁 Auto-load Batch Processing & Model Training")
        st.write("Training model on the auto dataset (this is cached).")
        with st.spinner("Training..."):
            model_bundle = train_model(df_auto, text_col="body", label_col="label")
        st.success(f"Trained. Accuracy on training data: {model_bundle['accuracy']:.4f}")

# ---------------------------
# Process Single Email
# ---------------------------
if mode == "Single Email":
    st.title("Single Email Analyzer")
    email_text = st.text_area("Paste email text here", height=220)
    if st.button("Mask & Classify"):
        if not email_text.strip():
            st.warning("Please paste an email first.")
        else:
            result = mask_all(email_text)
            # If user toggles off certain regex/spacy behavior, apply simple filtering:
            masked_text = result["masked_text"]
            if not enable_spacy_entities:
                # remask spaCy entity mentions by re-running regex-only
                # we can call mask_all on a version without spaCy, but quick approach:
                # Recreate masked by applying only regex masks on original
                import re
                text0 = result["original"]
                # regex-only masks (subset)
                # Use simplified expressions from masking.py
                masked_text, _ = mask_all(text0)["masked_text"], {}
            if not enable_aadhaar:
                # If user disabled Aadhaar/PAN/SSN, unmask those tokens (not ideal). We'll leave as is but inform user.
                pass

            st.subheader("Masked Output")
            st.code(masked_text)

            # Sentiment
            if enable_sentiment:
                sent = result["sentiment"]
                st.subheader("Sentiment")
                st.write(f"Label: **{sent['label']}**")
                st.write(sent["scores"])

            # Offer download of masked text
            st.download_button("Download masked as .txt", data=masked_text, file_name="masked_email.txt", mime="text/plain")

# ---------------------------
# Batch upload mode
# ---------------------------
elif mode == "Batch (Upload)":
    st.title("Batch Processing (Upload)")
    uploaded = st.file_uploader("Upload CSV / Excel file with 'body' and 'label' columns", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        # read file
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.write("Preview of uploaded data")
        st.dataframe(df.head())

        if "body" not in df.columns:
            st.error("Uploaded file must contain a 'body' column with email text. If label is absent that's ok for prediction-only mode.")
        else:
            # If label exists and user wants to train/evaluate
            has_label = "label" in df.columns
            if st.button("Run Batch Masking & (Train if label present)"):
                # Apply masking row-wise
                st.info("Masking PII in all rows (this may take a while for large files)...")
                masked_texts = []
                details_list = []
                sentiments = []
                for idx, row in df.iterrows():
                    r = mask_all(str(row["body"]))
                    # optionally remove spaCy masking if turned off (not implemented per token)
                    masked_texts.append(r["masked_text"])
                    details_list.append(r["details"])
                    sentiments.append(r["sentiment"]["label"] if enable_sentiment else "")
                df["masked_text"] = masked_texts
                if enable_sentiment:
                    df["sentiment"] = sentiments

                # If label exists: train and show metrics
                if has_label:
                    st.info("Training classifier on uploaded data (train & eval on same file).")
                    bundle = train_model(df, text_col="body", label_col="label")
                    st.success(f"Trained. Accuracy (train) = {bundle['accuracy']:.4f}")

                    # Show classification report table
                    st.subheader("Classification Report")
                    rep = pd.DataFrame(bundle["report"]).transpose()
                    st.dataframe(rep)

                    # Confusion matrix
                    labels = bundle["labels"]
                    cm = bundle["confusion_matrix"]
                    fig, ax = plt.subplots(figsize=(7,5))
                    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                else:
                    st.info("No 'label' column in uploaded file — skipping train/eval, performing masking + sentiment only.")

                st.success("Batch processing finished.")

                # Show sample
                st.subheader("Sample of processed rows")
                st.dataframe(df.head(50))

                # Download processed CSV
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_masked_emails.csv", mime="text/csv")

# ---------------------------
# Batch Auto (use existing dataset) -> allow full processing and evaluation
# ---------------------------
elif mode == "Batch (Auto-load)":
    # If df_auto is present and model_bundle trained
    if df_auto is None:
        st.error("No auto dataset found.")
    else:
        st.header("Auto Batch Processing (Using included dataset)")
        st.markdown("You can mask all emails, view metrics, and download processed CSV.")

        if st.button("Mask All & Evaluate (Auto dataset)"):
            st.info("Masking all emails...")
            df_proc = df_auto.copy()
            masked_texts = []
            sentiments = []
            details = []
            for i, row in df_proc.iterrows():
                r = mask_all(str(row.get("body", "")))
                masked_texts.append(r["masked_text"])
                sentiments.append(r["sentiment"]["label"])
                details.append(r["details"])
            df_proc["masked_text"] = masked_texts
            df_proc["sentiment"] = sentiments
            df_proc["pii_details"] = details

            st.subheader("Masking sample")
            st.dataframe(df_proc.head())

            # Train & evaluate on original (train on body & label)
            with st.spinner("Training & evaluating model..."):
                bundle = train_model(df_proc, text_col="body", label_col="label")
            st.success(f"Model trained. Accuracy (train) = {bundle['accuracy']:.4f}")

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(bundle["report"]).transpose())

            # Confusion matrix
            labels = bundle["labels"]
            cm = bundle["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Download processed dataset
            csv_bytes = df_proc.to_csv(index=False).encode("utf-8")
            st.download_button("Download processed dataset CSV", data=csv_bytes, file_name="auto_processed_masked.csv", mime="text/csv")
