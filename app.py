import streamlit as st
import numpy as np
import pandas as pd 
import pickle
from PIL import Image 

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


def main():
    activities=['About','Monitoring System','Developers']
    option=st.sidebar.selectbox('Menu Bar:',activities)
    if option=='About':
        html_temp = """
        <div style = "background-color: yellow; padding: 10px;">
            <center><h1>ABOUT OUR PROJECT</h1></center>
        </div><br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.title("What is Money Laundering?")
        image=Image.open('images/1.png')
        st.image(image, use_container_width=True)
        st.subheader('Money laundering is the illegal process of concealing the origins of money obtained illegally by passing it through a complex sequence of banking transfers or commercial transactions.')
        st.title('What is Transaction Monitoring?')
        st.subheader('Anti-money laundering (AML) transaction monitoring software allows banks and other financial institutions to monitor customer transactions on a daily basis or in real-time for risk.')
        st.subheader('By combining this information with analysis of customers’ historical information and account profile, the software can provide financial institutions with a “whole picture” analysis of a customer’s profile, risk levels, and predicted future activity, and can also generate reports and create alerts to suspicious activity')
        st.title('Our Product')
        st.subheader('Working of the product in a nutshell')
        image=Image.open('images/2.png')
        st.image(image, use_container_width=True)
        st.title('The future prospects ')
        st.subheader('With increasing crony capitalism and corruption due to the slightest of inefficiencies and people’s urges to make the most money in the shortest period of time, even not considering the legal implications, it is almost certain that we will see an increase in the amount of money being laundered in the global economy, without an interference by major players like financial institutions, consulting firms etc. ')
        st.subheader('To make our project self sustainable and ever adapting according to the market scenario, an integration of a simple Artificial neural network will be done to the model created, so that, as datasets become larger, the efficiency does not take a hit, and the model learns from itself, just like all of us. ')
    elif option=='Monitoring System':
        st.title("TRANSACTION MONITORING SYSTEM")
        image=Image.open('images/Magnify_Monitoring.jpg')
        st.image(image, use_container_width=True)
        st.subheader("Please Upload Your Dataset")
        data=st.file_uploader("Upload your dataset",type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
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
