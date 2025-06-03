import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# CPU Based Models
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Preprocessing & Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# Page configuration
st.set_page_config(
    page_title="Hoax Detection App",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    dataset = pd.read_csv("https://raw.githubusercontent.com/ghazimuharam/nlp-hoax-recognition/master/Data%20Latih/Data%20Latih%20BDC.csv")
    clean_dataset = pd.read_csv('preprocessed_data.csv') 
    return clean_dataset, dataset

clean_dataset, raw_dataset = load_data()
smote = SMOTE(random_state=42)
vectorizer = TfidfVectorizer()

x_vect = vectorizer.fit_transform(clean_dataset['text'])
x_train, x_test, y_train, y_test = train_test_split(x_vect, clean_dataset['label'], test_size=0.2, stratify=clean_dataset['label'], random_state=0)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Random Forest': RandomForestClassifier(max_depth=20, n_estimators=100),
    'SVC': SVC(C=10, kernel='rbf', probability=True),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'BernoulliNB': BernoulliNB(alpha=1e-9),
    'Logistic Regression': LogisticRegression(C=1),
    'KNN': KNeighborsClassifier(metric='euclidean', n_neighbors=3),
    'Ridge': RidgeClassifier(),
    'Passive Aggressive': PassiveAggressiveClassifier()
}

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

page = option_menu(
    menu_title=None,
    options=["Home", "EDA", "Stacking & Voting Model", "Model Training & Evaluation"],
    icons=["house", "bar-chart", "diagram-3", "robot"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "#4b6cb7", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#e9ecef"},
        "nav-link-selected": {"background-color": "#4b6cb7"},
    }
)

##################################################################################################################################################
if page == "Home":
    st.markdown('<h1 class="header-text">🕵️ Hoax Detection App</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:735/1*vNwJyRGHG-MyeYcCnuzXew.png", 
                width=500, 
                caption="Fake News Meme",
                use_column_width='auto')
    
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>👋 Welcome to Our Hoax Detection Platform!</h3>
            <p style='font-size: 1rem;'>
            This application helps detect <strong style='color: #4b6cb7;'>HOAXES</strong> in Indonesian Social Media text using advanced 
            <strong style='color: #4b6cb7;'>Stacking & Voting Ensemble Machine Learning</strong> techniques. 
            Developed as part of our Machine Learning course project under the guidance of Pak Johannes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="card">
        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🔍 Explore Our Features</h3>
        <ul style='margin-right:1rem; font-size: 1rem;'>
            <li>
                Explanatory Dataset
            </li>
            <li>
                Interactive Model Selection & Trainning
            </li>
            <li>
                Evaluation Metrics
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    features = [
        {"icon": "📊", "title": "EDA", "desc": "Explore the dataset with interactive visualizations"},
        {"icon": "🧩", "title": "Stacking & Voting", "desc": "Learn about ensemble techniques"},
        {"icon": "🤖", "title": "Model Training", "desc": "Train and evaluate ML models"},
        {"icon": "📈", "title": "Performance", "desc": "Compare model metrics"}
    ]
    
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{feature['icon']}</div>
                <h4 style='margin: 0; color: #2c3e50;'>{feature['title']}</h4>
                <p style='font-size: 0.8rem; margin: 0.5rem 0 0; color: #7f8c8d;'>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # How to use section
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📘 Getting Started Guide</h3>
            <ol style='padding-left: 1.2rem;'>
                <li style='margin-bottom: 0.5rem;'>Visit <strong>EDA</strong> to explore the dataset</li>
                <li style='margin-bottom: 0.5rem;'>Check <strong>Stacking & Voting</strong> to understand the methodology</li>
                <li style='margin-bottom: 0.5rem;'>Go to <strong>Model Training</strong> to build your own models</li>
                <li>Compare results and analyze performance metrics</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Inspirational quote
    st.markdown("""
    <div class="quote">
        "The best way to detect hate is to teach machines how to recognize it. So that we dont have to work."
        <div style='text-align: right; font-weight: bold; margin-top: 0.1rem;'>— Machine Learning Ethics</div>
    </div>
    """, unsafe_allow_html=True)

##################################################################################################################################################
elif page == "EDA":
    # Dataset Explanation Page
    st.markdown('<h1 class="header-text">📊 Dataset Explanation & EDA</h1>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1: 
            st.markdown("""
            <div class="card">
                <h3 style='color: #2c3e50;'>About the Dataset</h3>
                <p>This app uses a dataset for <strong style='color: #4b6cb7;'>Hoax Detection</strong> in Indonesian text.</p>
                <ul style='padding-left: 1.2rem;'>
                    <li>The original dataset <code>Data Latih BDC.csv</code> contains raw tweets with associated labels</li>
                    <li>The <code>preprocessed_data.csv</code> contains cleaned text for model training</li>
                    <li>The dataset has a combination of English & Indonesian Words</li>
                    <li>The dataset has many shortform text</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2: 
            st.markdown("""
            <div class="card">
                <h3 style='color: #2c3e50; margin-top: 1rem;'>Label Explanation:</h3>
                <div style='display: flex; gap: 1rem; margin-top: 0.5rem;'>
                    <span style='background-color: #e3f2fd; padding: 0.25rem 0.5rem; border-radius: 4px;'>0: Non-hoax</span>
                    <span style='background-color: #e8f5e9; padding: 0.25rem 0.5rem; border-radius: 4px;'>1: Hoax</span>
                </div>
                <h3 style='color: #2c3e50; margin-top: 1rem;'>Dataset Source:</h3>
                <p>This app uses a dataset from <a href="https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews">Kaggle</a></p> 
            </div>
            """, unsafe_allow_html=True)
    
    # Dataset Previews Section
    st.markdown("""
    <div class="card">
        <h2>Dataset Explorer</h2>
    </div>
    """, unsafe_allow_html=True)

    # Toggle between raw and clean datasets
    view_option = st.radio(
        "Select Dataset View",
        options=["Raw", "Clean"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Display dataset based on selection
    dataset_container = st.container()
    with dataset_container:
        if view_option == "Raw":
            st.markdown("""
            <div class="card" style="padding-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3 style='color: #2c3e50; margin: 0;'>Raw Dataset Preview</h3>
                    <span style="background-color: #ffebee; color: #c62828; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                        Unprocessed Data
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(raw_dataset, use_container_width=True,height=300,hide_index=True)
        else:
            st.markdown("""
            <div class="card" style="padding-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3 style='color: #2c3e50; margin: 0;'>Cleaned Dataset Preview</h3>
                    <span style="background-color: #e8f5e9; color: #2e7d32; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                        Processed Data
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(clean_dataset, use_container_width=True, height=300, hide_index=True)

    # Preprocessing Steps Section
    st.markdown("""
    <div class="card">
        <h2>
            <span style="display: inline-block; margin-right: 0.5rem;">🔧</span>
            Text Preprocessing Steps
        </h2>
        <p style="color: #666; margin-top: 0;">Explore the text cleaning and processing steps applied to the raw data, so that the model can receive it</p>
    </div>
    """, unsafe_allow_html=True)

    # Preprocessing steps with styled expanders
    steps = [
        {
            "title": "1. Lower Casing",
            "icon": "🔠",
            "explanation": "Converts all text to lowercase to maintain consistency (e.g., 'Jokowi' → 'jokowi').",
            "code": "text = text.lower()",
        },
        {
            "title": "2. Remove Punctuation",
            "icon": "❌",
            "explanation": "Removes punctuation like commas, periods, and other special characters.",
            "code": """
                import string
                text = text.translate(str.maketrans('', '', string.punctuation))
            """
        },
        {
            "title": "3. Remove Stopwords",
            "icon": "🚫",
            "explanation": "Removes commonly used words that do not carry significant meaning, like 'dan', 'yang', etc.",
            "code": """
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('indonesian'))
                text = ' '.join([word for word in text.split() if word not in stop_words])
            """
        },
        {
            "title": "4. Stemming",
            "icon": "✂️",
            "explanation": "Reduces words to their root form. For example, 'berjalan', 'berjalanlah', and 'berjalanannya' all become 'jalan'.",
            "code": """
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                text = stemmer.stem(text)
            """
        },
        {
            "title": "5. Vectorization",
            "icon": "🔢",
            "explanation": "Vectorizaion is the process of converting text data to numerical vectors. For this dataset we are going to use TF-IDF Vectorizer that converts a collection of raw document to a matrix to rank the relevance of a word in the entire document",
            "code": """
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                stemmer = factory.create_stemmer()
                x_vect = vectorizer.fit_transform(x)
            """
        }
    ]

    for step in steps:
        with st.expander(f"{step['icon']} {step['title']}"):
            st.markdown(f"""
            <div style="background-color: #e3f2fd20; padding: 1rem; border-radius: 8px; border-left: 4px solid #e3f2fd; margin-bottom: 1rem;">
                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{step['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Implementation:**")
            st.code(step['code'], language='python')
    
##################################################################################################################################################
elif page == "Stacking & Voting Model":
    # Stacking & Voting Explanation Page
    st.markdown('<h1 class="header-text">⚙️ Stacking & Voting Explanation</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Ensemble Learning Techniques</h3>
            <p>Ensemble learning in machine learning combines multiple individual models (weak learners) to create a stronger and more accurate predictive model. This approach leverages the diverse strengths of different models to improve performance, reduce errors, and increase robustness. Instead of relying on a single model's prediction, ensemble methods aggregate the predictions of multiple models to make a final prediction. </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Explanation Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card" style="height: 100%; min-height:45vh">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🧩</div>
            <h3 style='color: #2c3e50; margin: 0 0 0.5rem 0;'>What is Stacking?</h3>
            <p style='font-size: 1rem; margin: 0;'>
                Stacking is a technique that boosts AI capabilities by combining multiple machine learning models into one system. It works by taking the predictions from each model and feeding them into a final "meta-model" that learns how to best blend and stack their strengths.
            </p>
            <div style='margin-top: 1rem; background-color: #f5f5f5; padding: 0.75rem; border-radius: 6px;'>
                <ul style='margin: 0; padding-left: 1.2rem; font-size: 0.85rem;'>
                    <li>Base Models make initial predictions</li>
                    <li>A <strong>meta-model</strong> learns from these predictions</li>
                    <li>Often uses Logistic Regression as final estimator</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>How Stacking Works</h3>
            <div style='text-align: center; margin: 1rem 0; min-height:51vh; flex-direction:flex; align-item:center'>
                <img src='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*DM1DhgvG3UCEZTF-Ev5Q-A.png' style='max-width: 100%; border-radius: 8px;' alt='Stacking diagram'>
            </div>
            <p style='text-align: center; font-size: 0.85rem; color: #7f8c8d;'>
            Diagram showing the stacking ensemble workflow with base models and meta-model
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="height: 100%; min-height:45vh;">
            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🗳️</div>
            <h3 style='color: #2c3e50; margin: 0 0 0.5rem 0;'>What is Voting?</h3>
            <p style='font-size: 1rem; margin: 0;'>
            Voting Classifier is an ensemble learning technique that combines multiple classifiers and predicts the class based on a voting mechanism. This approach enhances model accuracy, reduces overfitting, and makes predictions more robust.
            </p>
            <div style='margin-top: 1rem; background-color: #f5f5f5; padding: 0.75rem; border-radius: 6px;'>
                <ul style='margin: 0; padding-left: 1.2rem; font-size: 0.85rem;'>
                    <li><strong>Hard voting</strong>: Majority class among models</li>
                    <li><strong>Soft voting</strong>: Uses predicted probabilities</li>
                    <li>Works best with diverse models</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>How Stacking Works</h3>
            <div style='text-align: center; margin: 1rem 0;'>
                <img src='https://miro.medium.com/v2/resize:fit:1400/1*djKLooxyOLvr98oMi5uwgA.jpeg' style='max-width: 100%; border-radius: 8px; ' alt='Stacking diagram'>
            </div>
            <p style='text-align: center; font-size: 0.85rem; color: #7f8c8d;'>
            Diagram showing the voting ensemble workflow with base models and voting output
            </p>
        </div>
        """, unsafe_allow_html=True)

##################################################################################################################################################
elif page == "Model Training & Evaluation":
    # Page Header
    st.markdown('<h1 class="header-text">🧠 Model Training & Evaluation</h1>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Build Your Ensemble Model</h3>
            <p>Select base models and voting strategy to evaluate performance.</p>
        </div>
        """, unsafe_allow_html=True)

    # Select models and voting strategy
    st.markdown("<h3 style='color: #2c3e50;'>Model Configuration</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect("Select Base Models", options=list(models.keys()), help="Choose at least two models")
    with col2:
        voting_strategy = st.selectbox("Voting Strategy", options=["soft", "hard"], help="Soft uses predicted probabilities")

    # Name for model combo (so user can select different trained stacks later)
    model_name = "_".join(selected_models)

    # Store all trained models in session
    if "trained_stacks" not in st.session_state:
        st.session_state.trained_stacks = {}

    # Train the model
    if len(selected_models) >= 2:
        if st.button("🚀 Train Stacking Model"):
            with st.spinner("Training the stacking model..."):
                base_estimators = [(name, models[name]) for name in selected_models if name != 'Logistic Regression']
                final_estimator = models["Logistic Regression"]

                stack_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    voting=voting_strategy,
                    n_jobs=-1
                )
                stack_model.fit(x_train_resampled, y_train_resampled)
                st.session_state.trained_stacks[model_name] = {
                    "model": stack_model,
                    "selected_models": selected_models,
                    "voting": voting_strategy
                }
            st.success(f"✅ '{model_name}' trained with '{voting_strategy}' voting!")

    # Choose trained model to evaluate
    if st.session_state.trained_stacks:
        st.markdown("### 🔄 Evaluate a Trained Stacking Model")
        chosen_stack = st.selectbox("Select a trained model", options=list(st.session_state.trained_stacks.keys()))
        chosen_voting = st.radio("Voting type to apply", options=["soft", "hard"], index=0)

        stored = st.session_state.trained_stacks[chosen_stack]

        if stored["voting"] != chosen_voting:
            st.info("⚙️ Retraining with new voting strategy...")
            base_estimators = [(name, models[name]) for name in stored["selected_models"] if name != 'Logistic Regression']
            final_estimator = models["Logistic Regression"]
            stack_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=final_estimator,
                voting=chosen_voting,
                n_jobs=-1
            )
            stack_model.fit(x_train_resampled, y_train_resampled)
        else:
            stack_model = stored["model"]

        y_pred = stack_model.predict(x_test)

        # Confusion Matrix
        st.markdown("#### 📊 Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.markdown("#### 📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Precision (Hoax)", f"{report['1']['precision']:.4f}")
        with col3:
            st.metric("Recall (Hoax)", f"{report['1']['recall']:.4f}")
    else:
        st.info("⚠️ No stacking model trained yet. Train one to evaluate.")