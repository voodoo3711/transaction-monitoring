import streamlit as st
import numpy as np
import pandas as pd 
import pickle
import warnings
from PIL import Image 

st.set_page_config(
    page_title="Transaction Monitoring",
    layout="wide",
)

MODEL_FEATURES = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
]

REQUIRED_COLUMNS = [
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
]

NUMERIC_COLUMNS = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
]

VALID_TRANSACTION_TYPES = {
    'CASH_IN',
    'CASH_OUT',
    'DEBIT',
    'PAYMENT',
    'TRANSFER',
}

PREDICTION_LABELS = {
    0: 'Normal',
    1: 'Suspicious',
}


@st.cache_resource(show_spinner=False)
def load_model(path='final_model.pkl'):
    try:
        model = load_pickle_model(path)
        return make_model_compatible(model)
    except ValueError as error:
        if 'missing_go_to_left' not in str(error):
            raise

    import sklearn.tree._tree as tree

    old_check_node_ndarray = tree._check_node_ndarray
    expected_dtype = tree.NODE_DTYPE
    old_node_fields = (
        'left_child',
        'right_child',
        'feature',
        'threshold',
        'impurity',
        'n_node_samples',
        'weighted_n_node_samples',
    )

    def check_node_ndarray(nodes, expected_dtype):
        if getattr(nodes, 'dtype', None) is not None and nodes.dtype.names == old_node_fields:
            converted = np.zeros(nodes.shape, dtype=expected_dtype)
            for field in old_node_fields:
                converted[field] = nodes[field]
            return converted
        return old_check_node_ndarray(nodes, expected_dtype)

    tree._check_node_ndarray = check_node_ndarray
    try:
        model = load_pickle_model(path)
        return make_model_compatible(model)
    finally:
        tree._check_node_ndarray = old_check_node_ndarray


def load_pickle_model(path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Trying to unpickle estimator.*",
        )
        with open(path, 'rb') as model_file:
            return pickle.load(model_file)


def make_model_compatible(model):
    if hasattr(model, 'base_estimator_') and not hasattr(model, 'estimator'):
        model.estimator = model.base_estimator_
    for estimator in getattr(model, 'estimators_', []):
        if not hasattr(estimator, 'monotonic_cst'):
            estimator.monotonic_cst = None
    return model


def preprocessData(dataFrame):
    dataFrame = dataFrame.drop(
        columns=['step', 'nameOrig', 'nameDest', 'y_prednew'],
        errors='ignore',
    )
    x_new = pd.get_dummies(dataFrame)
    return x_new.reindex(columns=MODEL_FEATURES, fill_value=0)


def validate_dataset(dataFrame):
    errors = []
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in dataFrame.columns
    ]
    if missing_columns:
        errors.append(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if dataFrame.empty:
        errors.append("The uploaded CSV does not contain any transaction rows.")

    for column in NUMERIC_COLUMNS:
        if column in dataFrame.columns:
            numeric_values = pd.to_numeric(dataFrame[column], errors='coerce')
            if numeric_values.isna().any():
                errors.append(f"Column '{column}' must contain only numeric values.")

    if 'type' in dataFrame.columns:
        if dataFrame['type'].isna().any():
            errors.append("Column 'type' must not contain blank values.")
        uploaded_types = set(dataFrame['type'].dropna().astype(str).str.strip().str.upper())
        if '' in uploaded_types:
            errors.append("Column 'type' must not contain blank values.")
            uploaded_types.remove('')
        invalid_types = sorted(uploaded_types - VALID_TRANSACTION_TYPES)
        if invalid_types:
            errors.append(
                "Unsupported transaction types: " + ", ".join(invalid_types)
            )

    return errors


def clean_dataset(dataFrame):
    cleaned = dataFrame.copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors='coerce')
    if 'type' in cleaned.columns:
        cleaned['type'] = cleaned['type'].astype(str).str.strip().str.upper()
    return cleaned


def render_about_page():
    st.markdown(
        """
        <style>
            .about-hero {
                border-radius: 8px;
                padding: 32px 34px;
                background: linear-gradient(135deg, #101820 0%, #1d2630 52%, #22313f 100%);
                border-left: 6px solid #f2c94c;
                margin-bottom: 22px;
            }
            .about-eyebrow {
                color: #f2c94c;
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 8px;
            }
            .about-title {
                color: #ffffff;
                font-size: 2.55rem;
                line-height: 1.1;
                font-weight: 800;
                margin: 0 0 12px 0;
            }
            .about-copy {
                color: #d8e2ec;
                font-size: 1.06rem;
                line-height: 1.65;
                max-width: 880px;
                margin: 0;
            }
            .stat-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
                margin: 18px 0 30px 0;
            }
            .stat-box {
                border-radius: 8px;
                padding: 18px;
                background: #ffffff;
                border: 1px solid #e7edf3;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }
            .stat-value {
                color: #eb5757;
                font-size: 1.55rem;
                font-weight: 800;
                margin-bottom: 4px;
            }
            .stat-label {
                color: #334155;
                font-size: 0.98rem;
                line-height: 1.45;
            }
            .section-title {
                color: #102a43;
                font-size: 1.55rem;
                font-weight: 800;
                margin: 8px 0 10px 0;
            }
            .section-copy {
                color: #334155;
                font-size: 1.02rem;
                line-height: 1.65;
                margin-bottom: 14px;
            }
            .accent-list {
                border-left: 4px solid #2d9cdb;
                padding: 6px 0 6px 16px;
                color: #334155;
                line-height: 1.75;
                margin-top: 10px;
            }
            .workflow {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 10px;
            }
            .workflow-step {
                border-radius: 8px;
                padding: 16px;
                background: #f8fafc;
                border-top: 4px solid #27ae60;
                color: #334155;
                min-height: 132px;
            }
            .workflow-step strong {
                display: block;
                color: #102a43;
                margin-bottom: 8px;
            }
            @media (max-width: 900px) {
                .stat-row,
                .workflow {
                    grid-template-columns: 1fr;
                }
                .about-title {
                    font-size: 2rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-hero">
            <div class="about-eyebrow">Final Year Project</div>
            <h1 class="about-title">Transaction Monitoring</h1>
            <p class="about-copy">
                An Anti-Money Laundering system that reviews transaction patterns and flags
                records that may need closer inspection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">AML</div>
                <div class="stat-label">Built around anti-money laundering transaction review.</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">CSV</div>
                <div class="stat-label">Accepts transaction datasets uploaded by the user.</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">Labels</div>
                <div class="stat-label">Classifies each record as Normal or Suspicious.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="section-title">What is money laundering?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-copy">
                Money laundering is the illegal process of hiding the origin of money
                obtained through unlawful activity. Funds are moved through transfers,
                payments, accounts, and commercial transactions to make them appear legitimate.
            </div>
            <div class="accent-list">
                The goal of this project is to help identify transactions that may deserve
                extra review before they become bigger compliance risks.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.image(Image.open('images/1.png'), use_container_width=True)

    st.markdown("---")

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.image(Image.open('images/2.png'), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">How transaction monitoring helps</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-copy">
                Transaction monitoring checks customer activity against known risk signals.
                The model uses transaction amount, balance movement, destination balance,
                and transaction type to classify each record.
            </div>
            <div class="accent-list">
                The output columns show both the model value and a readable label:
                <strong>Normal</strong> or <strong>Suspicious</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Project workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step">
                <strong>Upload</strong>
                Load a CSV containing transaction records.
            </div>
            <div class="workflow-step">
                <strong>Prepare</strong>
                Remove identifiers and encode the transaction type.
            </div>
            <div class="workflow-step">
                <strong>Predict</strong>
                Run the trained Random Forest model on each row.
            </div>
            <div class="workflow-step">
                <strong>Review</strong>
                Read the result table with the added prediction column.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_monitoring_page():
    st.markdown(
        """
        <style>
            .monitor-hero {
                border-radius: 8px;
                padding: 30px 32px;
                background: linear-gradient(135deg, #102a43 0%, #1f4e5f 58%, #2f6f73 100%);
                border-left: 6px solid #27ae60;
                margin-bottom: 20px;
            }
            .monitor-eyebrow {
                color: #a7f3d0;
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 8px;
            }
            .monitor-title {
                color: #ffffff;
                font-size: 2.35rem;
                line-height: 1.12;
                font-weight: 800;
                margin: 0 0 10px 0;
            }
            .monitor-copy {
                color: #e6fffb;
                font-size: 1.03rem;
                line-height: 1.62;
                max-width: 860px;
                margin: 0;
            }
            .upload-panel {
                border-radius: 8px;
                padding: 18px;
                background: #ffffff;
                border: 1px solid #dce8f2;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
                margin-bottom: 18px;
            }
            .panel-title {
                color: #102a43;
                font-size: 1.25rem;
                font-weight: 800;
                margin-bottom: 8px;
            }
            .panel-copy {
                color: #475569;
                line-height: 1.55;
                margin-bottom: 6px;
            }
            .result-note {
                border-left: 4px solid #27ae60;
                padding: 10px 0 10px 14px;
                color: #334155;
                background: #f8fafc;
                margin: 12px 0 18px 0;
            }
            .empty-state {
                border-radius: 8px;
                padding: 18px;
                background: #f8fafc;
                border: 1px dashed #b8c7d5;
                color: #475569;
                line-height: 1.6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="monitor-hero">
            <div class="monitor-eyebrow">Live prediction workspace</div>
            <h1 class="monitor-title">Transaction Monitoring System</h1>
            <p class="monitor-copy">
                Upload a transaction CSV, review the records, and run the trained model to
                flag transactions that may require AML review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro, visual = st.columns([1.08, 0.92], gap="large")
    with intro:
        st.markdown(
            """
            <div class="upload-panel">
                <div class="panel-title">Dataset upload</div>
                <div class="panel-copy">
                    Use a CSV with transaction amount, balance fields, account names,
                    destination details, and transaction type.
                </div>
                <div class="panel-copy">
                    The result includes <strong>y_prednew</strong> plus a readable
                    <strong>Prediction_Label</strong> column.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            with open('dummy_data.csv', 'rb') as sample_file:
                st.download_button(
                    "Download sample CSV",
                    data=sample_file.read(),
                    file_name="dummy_data.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            pass
        data = st.file_uploader(
            "Upload transaction CSV",
            type=['csv'],
            help="Upload a transaction dataset such as dummy_data.csv.",
        )
    with visual:
        st.image(Image.open('images/Magnify_Monitoring.jpg'), use_container_width=True)

    if data is None:
        st.markdown(
            """
            <div class="empty-state">
                Waiting for a CSV upload. Once uploaded, the table preview, dataset size,
                prediction button, and result summary will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.read_csv(data)
    validation_errors = validate_dataset(df)
    if validation_errors:
        st.error("This CSV cannot be used for prediction yet.")
        for error in validation_errors:
            st.warning(error)
        st.markdown(
            "Required columns: "
            + ", ".join(f"`{column}`" for column in REQUIRED_COLUMNS)
        )
        st.dataframe(df, use_container_width=True)
        return

    df = clean_dataset(df)
    st.success("Dataset successfully loaded")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", len(df))
    metric_cols[1].metric("Columns", len(df.columns))
    metric_cols[2].metric("Transaction types", df['type'].nunique() if 'type' in df.columns else "N/A")
    metric_cols[3].metric("Total amount", f"{df['amount'].sum():,.2f}" if 'amount' in df.columns else "N/A")

    st.markdown('<div class="panel-title">Uploaded dataset</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    if st.button("Run Prediction", type="primary"):
        model = load_model()
        x_new = preprocessData(df)
        result_df = df.copy()
        result_df['y_prednew'] = model.predict(x_new.to_numpy())
        result_df['Prediction_Label'] = (
            result_df['y_prednew']
            .map(PREDICTION_LABELS)
            .fillna('Unknown')
        )
        result_df.to_csv('final_result.csv', index=False)

        suspicious_count = int((result_df['y_prednew'] == 1).sum())
        normal_count = int((result_df['y_prednew'] == 0).sum())

        st.markdown(
            """
            <div class="result-note">
                Prediction complete. Review the full result table or focus on transactions
                marked as suspicious.
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_metrics = st.columns(3)
        result_metrics[0].metric("Normal", normal_count)
        result_metrics[1].metric("Suspicious", suspicious_count)
        result_metrics[2].metric("Output rows", len(result_df))

        all_rows, suspicious_rows = st.tabs(["All Results", "Suspicious Only"])
        with all_rows:
            st.dataframe(result_df, use_container_width=True)
        with suspicious_rows:
            st.dataframe(result_df[result_df['y_prednew'] == 1], use_container_width=True)

        st.download_button(
            "Download final_result.csv",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name="final_result.csv",
            mime="text/csv",
        )


def render_developers_page():
    st.markdown(
        """
        <style>
            .developer-hero {
                border-radius: 8px;
                padding: 34px;
                background: linear-gradient(135deg, #111827 0%, #263244 55%, #34495e 100%);
                border-left: 6px solid #2d9cdb;
                margin-bottom: 20px;
            }
            .developer-eyebrow {
                color: #93c5fd;
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 8px;
            }
            .developer-title {
                color: #ffffff;
                font-size: 2.45rem;
                line-height: 1.12;
                font-weight: 800;
                margin: 0 0 10px 0;
            }
            .developer-copy {
                color: #dbeafe;
                font-size: 1.05rem;
                line-height: 1.6;
                margin: 0;
            }
            .profile-grid {
                display: grid;
                grid-template-columns: 0.9fr 1.1fr;
                gap: 16px;
                margin-top: 16px;
            }
            .profile-card {
                border-radius: 8px;
                padding: 22px;
                background: #ffffff;
                border: 1px solid #e4edf6;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
                min-height: 210px;
            }
            .profile-mark {
                width: 74px;
                height: 74px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #102a43;
                color: #ffffff;
                font-size: 1.6rem;
                font-weight: 800;
                margin-bottom: 14px;
            }
            .profile-name {
                color: #102a43;
                font-size: 1.55rem;
                font-weight: 800;
                margin-bottom: 4px;
            }
            .profile-meta {
                color: #475569;
                line-height: 1.6;
            }
            .project-card {
                border-radius: 8px;
                padding: 20px;
                background: #f8fafc;
                border-left: 4px solid #eb5757;
                color: #334155;
                line-height: 1.65;
            }
            .project-card strong {
                color: #102a43;
            }
            .tag-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
            }
            .tag {
                border-radius: 999px;
                padding: 6px 12px;
                background: #eaf5ff;
                color: #0f4c81;
                font-size: 0.9rem;
                font-weight: 700;
            }
            @media (max-width: 900px) {
                .profile-grid {
                    grid-template-columns: 1fr;
                }
                .developer-title {
                    font-size: 2rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="developer-hero">
            <div class="developer-eyebrow">Project credit</div>
            <h1 class="developer-title">Developer</h1>
            <p class="developer-copy">
                Built as an MCA final year project focused on transaction monitoring and
                anti-money laundering support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="profile-grid">
            <div class="profile-card">
                <div class="profile-mark">DP</div>
                <div class="profile-name">Devansh Punj</div>
                <div class="profile-meta">IITM MCA 2024-2026</div>
                <div class="profile-meta">Final Year Project</div>
            </div>
            <div class="profile-card">
                <div class="project-card">
                    <strong>Transaction Monitoring - An Anti-Money Laundering System</strong><br>
                    This project uses a trained machine learning model to classify transaction
                    records and help identify suspicious activity for further AML review.
                </div>
                <div class="tag-row">
                    <span class="tag">Streamlit</span>
                    <span class="tag">Machine Learning</span>
                    <span class="tag">AML</span>
                    <span class="tag">Random Forest</span>
                    <span class="tag">Python</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    activities=['About','Monitoring System','Developers']
    option=st.sidebar.selectbox('Menu Bar:',activities)
    if option=='About':
        render_about_page()
    elif option=='Monitoring System':
        render_monitoring_page()
    elif option=='Developers':
        render_developers_page()

if __name__=='__main__':
    main()
