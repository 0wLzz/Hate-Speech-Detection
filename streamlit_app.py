import time
import pickle
import os
import pandas as pd
import streamlit as st

# Preprocessing & Metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu

# Pages 
from sections.homepage import home_page
from sections.eda_page import eda_page
from sections.model_explanation import model_explanation_page
from sections.model_training import model_training_page 
from sections.detector_page import detector_page

# Page configuration
st.set_page_config(
    page_title="Hoax Detection App",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'active_page' not in st.session_state: 
    st.session_state.active_page = "Home"
if "trained_stacks" not in st.session_state:
    st.session_state.trained_stacks = {}

def load_initial_model():
    model_path = "voting_classifier.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                pretrained_model = pickle.load(f)
                st.session_state.trained_stacks["Pre-trained Voting Classifier"] = pretrained_model
        except Exception as e:
            st.error(f"Could not load the default model '{model_path}': {e}", icon="🚨")

# Run this only once, if no models are loaded yet
if not st.session_state.trained_stacks:
    load_initial_model()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def load():
    smote = SMOTE(random_state=42)
    vectorizer = TfidfVectorizer()

    dataset = pd.read_csv("https://raw.githubusercontent.com/ghazimuharam/nlp-hoax-recognition/master/Data%20Latih/Data%20Latih%20BDC.csv")
    clean_dataset = pd.read_csv('preprocessed_data.csv') 

    vectorizer.fit(clean_dataset['text'])

    x_text_train, x_text_test, y_train, y_test = train_test_split(clean_dataset['text'], clean_dataset['label'], test_size=0.2, stratify=clean_dataset['label'], random_state=0)

    x_train = vectorizer.transform(x_text_train)
    x_test = vectorizer.transform(x_text_test)

    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    # x_train_resampled = x_train_resampled.toarray() 

    return clean_dataset, dataset, x_train_resampled, y_train_resampled, x_text_test, x_test,  y_train, y_test, vectorizer

if __name__ == "__main__":
    # Load Dataset
    clean_dataset, raw_dataset, x_train_resampled, y_train_resampled, x_text_test, x_test, y_train, y_test, vectorizer = load()
    local_css("style.css")

    page_options_list = ["Home", "EDA", "Stacking & Voting Model", "Model Training & Evaluation", "Hoax Detector"]
    page_icons_list = ["house", "bar-chart", "diagram-3", "robot"]

    try:
        current_default_index = page_options_list.index(st.session_state.active_page)
    except ValueError:
        current_default_index = 0

    selected_page_by_menu = option_menu(
        menu_title=None,
        options=page_options_list,
        icons=page_icons_list,
        menu_icon="cast",
        default_index=current_default_index,
        orientation="horizontal",
        styles = {
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#4b6cb7", "font-size": "14px"},
            "nav-link": {"margin":"0px"},
            "nav-link-selected": {"background-color": "#4b6cb7", "color": "#f8f9fa"},
        }
    )

    if selected_page_by_menu != st.session_state.active_page:
        st.session_state.active_page = selected_page_by_menu
        st.rerun()

    if st.session_state.active_page == "Home":
        home_page()
    elif st.session_state.active_page == "EDA":
        eda_page(raw_dataset, clean_dataset)
    elif st.session_state.active_page == "Stacking & Voting Model":
        model_explanation_page()
    elif st.session_state.active_page == "Model Training & Evaluation":
       model_training_page(x_train_resampled, y_train_resampled, x_test, y_test, x_text_test)
    elif st.session_state.active_page == "Hoax Detector":
        detector_page(vectorizer)
