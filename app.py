import streamlit as st
import numpy as np
import pandas as pd 
import pickle
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


def load_model(path='final_model.pkl'):
    try:
        model = pickle.load(open(path, 'rb'))
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
        model = pickle.load(open(path, 'rb'))
        return make_model_compatible(model)
    finally:
        tree._check_node_ndarray = old_check_node_ndarray


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
                <div class="stat-value">0 / 1</div>
                <div class="stat-label">Classifies each record as normal or suspicious.</div>
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
                The output column <strong>y_prednew</strong> marks each transaction:
                <strong>0</strong> for normal and <strong>1</strong> for suspicious.
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


def main():
    activities=['About','Monitoring System','Developers']
    option=st.sidebar.selectbox('Menu Bar:',activities)
    if option=='About':
        render_about_page()
    elif option=='Monitoring System':
        st.title("TRANSACTION MONITORING SYSTEM")
        image=Image.open('images/Magnify_Monitoring.jpg')
        st.image(image, use_container_width=True)
        st.subheader("Please Upload Your Dataset")
        data=st.file_uploader("Upload your dataset",type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df)
            st.success("Data Successfully loaded")
            model = load_model()

            if st.button("Predict"):
                st.balloons()
                x_new = preprocessData(df)
                df['y_prednew'] = model.predict(x_new)
                df.to_csv('final_result.csv')

                st.title("Your Output is")
                st.dataframe(df)
    elif option=='Developers':
        st.title("Developers")
        st.header("Devansh Punj")
        st.subheader("IITM MCA 2024-2026")
        st.markdown("**Final Year Project**")
        st.markdown("**Transaction Monitoring - An Anti-Money Laundering System**")

                
            

if __name__=='__main__':
    main()
