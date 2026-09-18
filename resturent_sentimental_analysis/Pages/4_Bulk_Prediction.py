
import io

import pandas as pd
import streamlit as st

from Utils.loader import (
    load_pipeline,
    load_labels,
    load_css
)

# PAGE CONFIG

st.set_page_config(
    page_title="Bulk Sentiment Prediction",
    layout="wide"
)

load_css()

# LOAD FILES

pipeline = load_pipeline()
labels = load_labels()

# HEADER

st.markdown(
    '<div class="page-title bulk-pred-title">Bulk Review Sentiment Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Upload a file containing multiple reviews to predict sentiment for "
    "all of them at once, using the same preprocessing, TF-IDF and model "
    "pipeline as the single review predictor."
)

st.divider()


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

CANDIDATE_REVIEW_COLUMNS = [
    "review", "reviews", "review_text", "text", "comment",
    "comments", "feedback", "description", "message"
]


def read_uploaded_file(uploaded_file):
    """
    Try hard to read whatever the user uploaded (csv, tsv, txt, xlsx)
    without throwing an error back at the user. Falls back through a
    number of strategies and only returns None if the file truly has
    no readable tabular content.
    """

    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    # Excel files
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(raw_bytes))
        except Exception:
            pass

    # Try a series of separators for text-based files
    separators = ["\t", ",", ";", "|"]

    for sep in separators:
        try:
            df_try = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=sep,
                engine="python",
                on_bad_lines="skip"
            )
            # A real parse should give us more than one column,
            # or at least one usable column with more than one row.
            if df_try.shape[1] >= 1 and df_try.shape[0] > 0:
                return df_try
        except Exception:
            continue

    # Last resort: treat every non-empty line as a single review
    try:
        text = raw_bytes.decode("utf-8", errors="ignore")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() != ""
        ]
        if lines:
            return pd.DataFrame({"Review": lines})
    except Exception:
        pass

    return None


def detect_review_column(df):
    """
    Automatically guess which column holds the review text.
    Never raises - always returns a best-effort column name.
    """

    if df is None or df.shape[1] == 0:
        return None

    # 1. Exact / partial name match against known review-like names
    lower_cols = {col: str(col).strip().lower() for col in df.columns}

    for col, lower in lower_cols.items():
        if lower in CANDIDATE_REVIEW_COLUMNS:
            return col

    for col, lower in lower_cols.items():
        if any(candidate in lower for candidate in CANDIDATE_REVIEW_COLUMNS):
            return col

    # 2. Fall back to the text-like (object) column with the longest
    #    average text length - almost always the free-text review column.
    object_cols = df.select_dtypes(include="object").columns.tolist()

    if object_cols:
        avg_lengths = {
            col: df[col].astype(str).str.len().mean()
            for col in object_cols
        }
        return max(avg_lengths, key=avg_lengths.get)

    # 3. Absolute fallback - just use the first column.
    return df.columns[0]


def clean_review_series(series):
    """
    Make any input column safe to run through the pipeline:
    fills missing values, forces everything to string, strips
    whitespace, and never errors out regardless of what's inside.
    """

    cleaned = series.copy()

    cleaned = cleaned.where(cleaned.notna(), "")
    cleaned = cleaned.astype(str)
    cleaned = cleaned.str.strip()

    # Rows that became empty after cleaning are replaced with a
    # neutral placeholder so the pipeline never sees an empty string.
    cleaned = cleaned.replace("", "no review provided")
    cleaned = cleaned.replace("nan", "no review provided")
    cleaned = cleaned.replace("none", "no review provided")

    return cleaned


# ---------------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------------

st.markdown('<div class="section-title">1. Upload Reviews File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV, TSV or Excel file containing a review column",
    type=["csv", "tsv", "txt", "xlsx", "xls"]
)

raw_df = None

if uploaded_file is not None:
    raw_df = read_uploaded_file(uploaded_file)

    if raw_df is None:
        st.warning(
            "The file could not be read as a table, so it was loaded as "
            "plain text instead - one review per line."
        )

if raw_df is None:
    st.info(
        "Upload a file above to run bulk sentiment prediction."
    )
    st.stop()

# Drop fully empty rows/columns automatically instead of erroring later
raw_df = raw_df.dropna(axis=1, how="all")
raw_df = raw_df.dropna(axis=0, how="all").reset_index(drop=True)

if raw_df.shape[0] == 0 or raw_df.shape[1] == 0:
    st.warning(
        "The uploaded file appears to be empty. Please upload a file "
        "that contains at least one review."
    )
    st.stop()

st.success(
    f"File loaded successfully - {raw_df.shape[0]} rows, "
    f"{raw_df.shape[1]} columns."
)


# ---------------------------------------------------------------
# COLUMN SELECTION
# ---------------------------------------------------------------

st.markdown('<div class="section-title">2. Select Review Column</div>', unsafe_allow_html=True)

detected_column = detect_review_column(raw_df)
column_options = list(raw_df.columns)

default_index = (
    column_options.index(detected_column)
    if detected_column in column_options
    else 0
)

review_column = st.selectbox(
    "Column containing the review text "
    "(auto-detected, change if needed)",
    column_options,
    index=default_index
)


# ---------------------------------------------------------------
# PREVIEW (RAW)
# ---------------------------------------------------------------

st.markdown('<div class="section-title">3. Dataset Preview</div>', unsafe_allow_html=True)

preview_rows = st.slider(
    "Number of rows to preview",
    min_value=5,
    max_value=min(100, max(5, raw_df.shape[0])),
    value=min(10, raw_df.shape[0]),
    step=5
)

st.dataframe(
    raw_df.head(preview_rows),
    use_container_width=True
)


# ---------------------------------------------------------------
# BULK PREDICTION
# ---------------------------------------------------------------

st.divider()
st.markdown('<div class="section-title">4. Run Bulk Prediction</div>', unsafe_allow_html=True)

run_prediction = st.button(
    "Predict Sentiment for All Rows",
    use_container_width=True
)

if run_prediction:

    with st.spinner("Cleaning text and running predictions..."):

        working_df = raw_df.copy()

        try:
            # Same preprocessing / cleaning safety net as the rest
            # of the pipeline - never lets a bad row crash the app.
            cleaned_reviews = clean_review_series(
                working_df[review_column]
            )

            reviews_list = cleaned_reviews.tolist()

            # Same pipeline as single prediction: preprocessor -> tfidf -> model
            predictions = pipeline.predict(reviews_list)
            probabilities = pipeline.predict_proba(reviews_list)

            confidence = probabilities.max(axis=1) * 100

            sentiments = [labels[p] for p in predictions]

            result_df = working_df.copy()

            result_df["Predicted_Sentiment"] = sentiments
            result_df["Confidence_%"] = confidence.round(2)

        except Exception:

            # Absolute fallback: never surface a raw error to the user.
            # Re-run row-by-row so a single bad row can't block the rest.
            sentiments = []
            confidence = []

            for text in working_df[review_column].tolist():
                try:
                    safe_text = str(text).strip() or "no review provided"
                    pred = pipeline.predict([safe_text])[0]
                    proba = pipeline.predict_proba([safe_text])[0]
                except Exception:
                    pred = 0
                    proba = [1.0, 0.0]

                sentiments.append(labels.get(pred, "Unknown"))
                confidence.append(round(max(proba) * 100, 2))

            result_df = working_df.copy()
            result_df["Predicted_Sentiment"] = sentiments
            result_df["Confidence_%"] = confidence

        st.session_state["bulk_result_df"] = result_df

# ---------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------

if "bulk_result_df" in st.session_state:

    result_df = st.session_state["bulk_result_df"]

    st.divider()
    st.markdown('<div class="section-title">5. Prediction Results</div>', unsafe_allow_html=True)

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # Download results
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions as CSV",
        data=csv_bytes,
        file_name="bulk_prediction_results.csv",
        mime="text/csv",
        use_container_width=True
    )
