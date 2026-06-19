"""
upload_module.py
-------------------
Dataset Upload Module.

Lets a logged-in user either:
  * upload a real CSV (CICIDS2017, NSL-KDD, or any flow-style export), or
  * click "Load Sample Dataset" to instantly generate a synthetic
    CICIDS2017-style dataset (data_utils.generate_synthetic_dataset) so the
    whole pipeline is explorable without sourcing a multi-GB public dataset.

After a dataset is loaded it is:
  1. Cleaned (preprocessing.clean_dataframe)
  2. Profiled (rows / cols / missing / duplicates / class distribution)
  3. Persisted to disk + recorded in the `datasets` table
  4. Stashed in st.session_state["active_df"] / ["active_dataset_id"] /
     ["active_dataset_name"] so user_dashboard.py's "Analyze" pipeline can
     pick it up directly without re-uploading.

   # FUTURE SCOPE: stream large files in chunks instead of loading the
   # full CSV into memory, and add a connector for live packet-capture
   # ingestion (pcap -> flow features) instead of pre-exported CSVs.
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

import config
import data_utils
import database
import preprocessing


def _section_title(text, icon="📂"):
    st.markdown(
        f"<div class='ss-section-title'><span class='bar'></span>{icon} {text}</div>",
        unsafe_allow_html=True,
    )


def _persist_dataset(user_id, filename, df, label_col):
    """Save the cleaned dataframe to disk and record metadata in the DB."""
    stats = preprocessing.basic_dataset_stats(df, label_col)
    safe_name = f"{user_id}_{int(datetime.utcnow().timestamp())}_{filename}"
    stored_path = os.path.join(config.UPLOADS_DIR, safe_name)
    df.to_csv(stored_path, index=False)

    dataset_id = database.save_dataset(
        user_id=user_id,
        filename=filename,
        stored_path=stored_path,
        rows=stats["rows"],
        cols=stats["cols"],
        missing=stats["missing_values"],
        duplicates=stats["duplicate_values"],
        class_dist=stats.get("class_distribution", {}),
    )
    database.log_action(user_id, "dataset_uploaded", filename)
    return dataset_id, stats


def _show_dataset_summary(df, label_col, stats):
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        (c1, c2, c3, c4),
        ("Rows", "Columns", "Missing Values", "Duplicate Rows"),
        (stats["rows"], stats["cols"], stats["missing_values"], stats["duplicate_values"]),
    ):
        with col:
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{value:,}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("##### Preview")
    st.dataframe(df.head(15), use_container_width=True, height=320)

    if label_col and stats.get("class_distribution"):
        st.markdown("##### Class Distribution")
        dist_df = pd.DataFrame(
            list(stats["class_distribution"].items()), columns=["Class", "Count"]
        ).sort_values("Count", ascending=False)
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with cc2:
            try:
                import charts
                st.plotly_chart(
                    charts.attack_distribution_pie(stats["class_distribution"]),
                    use_container_width=True,
                )
            except Exception:
                st.bar_chart(dist_df.set_index("Class"))
    else:
        st.info("No label/attack-type column was detected in this dataset — that's fine, "
                 "the Deep Learning IDS will still classify each row using the trained model.")


def render_upload_page(user):
    _section_title("Dataset Upload", "📂")
    st.caption(
        "Upload a CICIDS2017 / NSL-KDD style CSV, or load an instant synthetic sample "
        "to explore the full detection pipeline."
    )

    tab_upload, tab_sample, tab_history = st.tabs(
        ["⬆️ Upload CSV", "🧪 Load Sample Dataset", "🕘 Upload History"]
    )

    # ---------------- Upload CSV ----------------
    with tab_upload:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read this file as CSV: {exc}")
                raw_df = None

            if raw_df is not None and not raw_df.empty:
                label_col = preprocessing.detect_label_column(raw_df)
                clean_df = preprocessing.clean_dataframe(raw_df)

                st.success(f"Loaded **{uploaded_file.name}** — {clean_df.shape[0]:,} rows × "
                           f"{clean_df.shape[1]:,} columns (after cleaning).")
                stats = preprocessing.basic_dataset_stats(clean_df, label_col)
                _show_dataset_summary(clean_df, label_col, stats)

                if st.button("✅ Use This Dataset for Analysis", type="primary"):
                    dataset_id, stats = _persist_dataset(user["id"], uploaded_file.name, clean_df, label_col)
                    st.session_state.active_df = clean_df
                    st.session_state.active_label_col = label_col
                    st.session_state.active_dataset_id = dataset_id
                    st.session_state.active_dataset_name = uploaded_file.name
                    st.success("Dataset saved. Head to **Threat Detection** to run the analysis.")

    # ---------------- Sample dataset ----------------
    with tab_sample:
        st.write(
            "Generates a realistic synthetic CICIDS2017-style flow dataset covering all "
            f"{len(config.ATTACK_CATEGORIES)} traffic classes: "
            + ", ".join(config.ATTACK_CATEGORIES)
        )
        n_per_class = st.slider("Rows per class", min_value=50, max_value=1000, value=200, step=50)
        if st.button("🧪 Generate Sample Dataset", type="primary"):
            sample_df = data_utils.generate_synthetic_dataset(n_per_class=n_per_class)
            label_col = data_utils.LABEL_COLUMN
            stats = preprocessing.basic_dataset_stats(sample_df, label_col)

            st.success(f"Generated {sample_df.shape[0]:,} synthetic rows.")
            _show_dataset_summary(sample_df, label_col, stats)

            dataset_id, stats = _persist_dataset(user["id"], "synthetic_sample.csv", sample_df, label_col)
            st.session_state.active_df = sample_df
            st.session_state.active_label_col = label_col
            st.session_state.active_dataset_id = dataset_id
            st.session_state.active_dataset_name = "synthetic_sample.csv"
            st.info("This sample is now your active dataset. Head to **Threat Detection** to analyze it.")

    # ---------------- History ----------------
    with tab_history:
        history = database.list_datasets(user_id=user["id"])
        if not history:
            st.info("No datasets uploaded yet.")
        else:
            hist_df = pd.DataFrame(history)[
                ["id", "filename", "rows", "cols", "missing_values", "duplicate_values", "uploaded_at"]
            ]
            hist_df.columns = ["ID", "Filename", "Rows", "Cols", "Missing", "Duplicates", "Uploaded At"]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
