# import libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import requests
import shap
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
import json
from explainerdashboard import InlineExplainer
import base64
import urllib.request

# CSS
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-color: rgba(255,255,255,0.8);
    background-blend-mode: lighten;
    background-size: cover;
    }
    .css-b7s55g{
        background: #08105c;
    }

    .css-1yjuwjr{
        font-size: 16px;
        font-weight:700;
        padding-left: 0.5rem;
    }

    .st-au {
    background-color: #08105c;
    color: #f3f9fd;
    text-align: center;

    .main-svg{
        background-color: #11ffee00 !important;
    }

    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

change_text = """
<style>
div.st-cs.st-c5.st-bc.st-ct.st-cu {visibility: hidden;}
div.st-cs.st-c5.st-bc.st-ct.st-cu:before {content: "Sélectionner l'information à afficher :"; visibility: visible;}
</style>
"""


def load_data():
    # load datas
    # data of the customers
    data = pd.read_csv('datas/customer_sample.csv.zip', index_col='SK_ID_CURR')

    # data of the customers preprocessed 
    sample = pd.read_csv('datas/sample_preproc.csv.zip', index_col='SK_ID_CURR')
    
    return data, sample

def load_explainer():
    
    # load shap_values
    shap_values = pickle.load(urllib.request.urlopen("https://www.dropbox.com/s/e4yj7wwnz0tzz4q/shap_values.pkl?dl=1"))

    return shap_values

def load_model():
    # load model
    pickle_classifier = open('models/LGBMClassifier.pkl','rb')
    clf=pickle.load(pickle_classifier)

    return clf

@st.cache
def load_age(data):
    data_age=round((data.DAYS_BIRTH/-365), 0)
    return data_age

def load_experience(data):
    data_employed=round((data.DAYS_EMPLOYED * -1 / 365), 0)
    return data_employed

# function to display shap plot as html
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

######################################################################################################################################## 

# load logo
logo = Image.open('datas/logo.png')

# thershold for custom score
threshold = 0.320

set_background('datas/bk.png')

col1, mid, col2 = st.columns([2,2,8])
with mid:
    st.image(logo)
with col2:
    st.markdown("<h1 style='color: #08105c;'>CREDIT SCORING</h1>", unsafe_allow_html=True)

#Loading data....
data, sample = load_data()
id_client = sample.index.values

clf = load_model()

# selectbox for customersID
customerid=st.sidebar.selectbox("Client ID", id_client)

# Info about client
customer_data = data[data.index==customerid].iloc[0]

# row of the customer in data
customer_row = data.index.get_loc(customerid)


# checkbox to display different options
cbx_proba = st.sidebar.button('Prédire')
cbx_data = st.sidebar.checkbox('Interprétabilié')
cbx_compare = st.sidebar.checkbox('Autre stats')

######################################################################################################################################## 

# client info 
st.sidebar.header("Information du client")
st.sidebar.write("CLIENT ID : ", str(customerid))
st.sidebar.write("GENRE : ", customer_data.CODE_GENDER)
st.sidebar.write("SITUATION : ", customer_data.NAME_FAMILY_STATUS)
st.sidebar.write("ÂGE : ", str(load_age(customer_data)))
st.sidebar.write("REVENUE TOTALE : ", str(customer_data.AMT_INCOME_TOTAL), "years")
st.sidebar.write("TYPE DE REVENUE : ", customer_data.NAME_INCOME_TYPE)
st.sidebar.write("ANCIENNETE : ", str(load_experience(customer_data)), "years")

# if cbx_proba checkbox checked
if cbx_proba:

    set_background('datas/bk2.png')
    
    api= requests.get("https://pretscore.herokuapp.com/predict/"+ str(customerid))
    response = api.json()

    st.info("Prédiction de solvalibité")
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = response["prediction"],
    title = {"text": "<span style='font-size:26px;font-weight:700;color:08105c'>" + response["decision"] + "</span>"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {'axis': {'range': [None, 100]},'bar': {'color': "powderblue"},
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold*100}}
    ))

    fig.update_layout(paper_bgcolor = "#f3f9fd", font = {'color': "#08105c"})

    st.plotly_chart(fig, use_container_width=True)
       
########################################################################################################################################     

# if cbx_data checkbox checked
if cbx_data:  
      
    st.info("Interprétabilité du défaut de paiement")
    set_background('datas/bk2.png')

    shap_values = load_explainer()

    explainers = shap.TreeExplainer(clf)
    data_for_prediction = sample[sample.index==customerid]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction.drop('Unnamed: 0', inplace=True, axis=1)
    

    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    # Calculate Shap values
    shap_value = explainers.shap_values(data_for_prediction_array)
    ind_fig = shap.force_plot(explainers.expected_value[1],
                    shap_value[1],
                    data_for_prediction,
                    plot_cmap=["#EF553B","#636EFA"])
    ind_fig_html = f"<head>{shap.getjs()}</head><body>{ind_fig.html()}</body>"
    components.html(ind_fig_html)
    
    # display graph from shap for customer selected
    
    st.info("Variables importantes du client")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.plots.waterfall(shap_values[customer_row])
    st.pyplot(fig)
        
    st.info("Variables importantes du modèle")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values)
    st.pyplot(fig)
        
########################################################################################################################################       
        
# if cbx_compare checkbox checked
if cbx_compare:

    st.markdown(change_text, unsafe_allow_html=True)

    #multiselect options to view differentt graphs

    features = st.multiselect(
                label="", options=['GENRE','ÂGE','STATUT','REVENUE TOTALE','TYPE DE REVENUE','ANCIENNETE'])
    
    set_background('datas/bk2.png')

    if "ÂGE" in features:

        st.info("Distribution des âges")
        data_age = load_age(data)
        fig = go.Figure()
        fig = go.Figure(data=[go.Histogram(x=data_age)])
        #fig.add_vline(x=int(load_age(customer_data)), line_dash = 'dash', line_color = '#f3f9fd')

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        xaxis_title_text='âge', # xaxis label
        yaxis_title_text='Nombre de clients', # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        )
        st.plotly_chart(fig)

    if "GENRE" in features:

        st.info("Distribution du genre")
        data_status = data.CODE_GENDER
        fig = go.Figure(data=[go.Pie(labels=data_status.unique(), values=data_status.value_counts())])

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        )
        st.plotly_chart(fig)

    if "TYPE DE REVENUE" in features:

        st.info("Distribution du type de revenue")
        fig = go.Figure(data=[go.Pie(labels=data.NAME_INCOME_TYPE.unique(), values=data.NAME_INCOME_TYPE.value_counts())])

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        )

        st.plotly_chart(fig)

    if "STATUT" in features:

        st.info("Distribution des statuts des clients")

        fig = go.Figure(data=[go.Pie(labels=data.NAME_FAMILY_STATUS.unique(), values=data.NAME_FAMILY_STATUS.value_counts())])

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        )

        st.plotly_chart(fig)

    if "REVENUE TOTALE" in features:

        st.info("Distribution du revenue totale")

        fig = go.Figure()
        fig = go.Figure(data=[go.Histogram(x=data.AMT_INCOME_TOTAL)])
        fig.add_vline(x=int(customer_data.AMT_INCOME_TOTAL), line_dash = 'dash', line_color = '#f3f9fd')

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        xaxis_title_text='Totale Revenue', # xaxis label
        yaxis_title_text='Nombre de clients', # yaxis label
        bargap=0.3, # gap between bars of adjacent location coordinates
        )

        st.plotly_chart(fig)

    if "ANCIENNETE" in features:

        st.info("Distribution d'ancienneté des clients")

        data_exp = load_experience(data)
        fig = go.Figure()
        fig = go.Figure(data=[go.Histogram(x=data_exp)])
        fig.add_vline(x=int(load_experience(customer_data)), line_dash = 'dash', line_color = '#f3f9fd')

        fig.update_layout(
        paper_bgcolor = "#f3f9fd", 
        font = {'color': "#08105c"},
        xaxis_title_text='Années', # xaxis label
        yaxis_title_text='Nombre de clients', # yaxis label
        )

        st.plotly_chart(fig)
