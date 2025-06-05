import time
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Load Model
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Random Forest': RandomForestClassifier(max_depth=20, n_estimators=100),
    'SVC': SVC(C=10, kernel='rbf'),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'BernoulliNB': BernoulliNB(alpha=1e-9),
    'Logistic Regression': LogisticRegression(C=1),
    'KNN': KNeighborsClassifier(metric='euclidean', n_neighbors=3),
    'Ridge': RidgeClassifier(),
    'Passive Aggressive': PassiveAggressiveClassifier()
}

def train_stack_model(stacks, base, x_train_resampled, y_train_resampled):
    base_estimators = [(name, models[name]) for name in stacks]
    final_estimator = models[base]

    stack_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        n_jobs=-1
    )

    stack_model.fit(x_train_resampled, y_train_resampled)

    return stack_model

@st.fragment()
def download_model(voting_ensamble):
    filename = st.text_input(
        label="What do you want to save it as?",
        max_chars=255,
        placeholder="Filename",
        key="model_filename"
    ) 

    if st.button("Download Model", use_container_width=True, type='primary') and filename:
        model_download = voting_ensamble
        
        file_path = f"{filename}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model_download, f)

        with open(file_path, "rb") as f:
            st.download_button(
                label="Click here to download your model",
                data=f,
                file_name=file_path,
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.warning("Please enter a filename before downloading.")


def model_training_page(x_train_resampled, y_train_resampled, x_test, y_test, x_text_test):
     # Page Header
    st.markdown('<h1 class="header-text">🧠 Model Training & Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('')

    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Build Your Ensemble Model</h3>
            <p>Select base models and voting strategy to evaluate performance.</p>
        </div>
        """, unsafe_allow_html=True)

    # Store all trained models in session
    if "trained_stacks" not in st.session_state:
        st.session_state.trained_stacks = {}

    num_stacks = st.slider(
        label="How Many Stacks Do you want to make? (Odd Number)", 
        step=2,
        min_value = 1,
        max_value = 9)
    
    st.markdown("<div style='text-align: center;' class='header-text'> Configure your Stack Models </div>", unsafe_allow_html=True) 
    col1, col2 = st.columns(2)

    with col1:
        with st.container(height=500, border=False):
            for num in range(num_stacks):
                st.header(f"Stack Model {num + 1}")
                st.multiselect(f"Select Meta Learners", models.keys(), key=f"stack{num}")

    with col2:
        st.header("Meta Learner")
        st.selectbox(
            label=f"Select Meta Learners", 
            options = models.keys(), 
            key="meta_learner"
            )
        st.header("Voting Strategy")
        vote_strategy = st.selectbox(
            label="Please choose your voting strategy!",
            options = ['Hard', 'Soft'],
            format_func = lambda x : x + "Voting",
            help = 'Hard Voting is Majority Classification, Soft combines the Probabilities',
            label_visibility='visible')

    # Train the model
    if st.button("Train All Stacks", use_container_width=True, type='primary'):
        progress_bar = st.progress(0, text='Trainning Stack Model(s) in Progress')
        for i in range(num_stacks):
            current_stack_selections = st.session_state.get(f"stack{i}", [])
            meta_learner = st.session_state.get('meta_learner', '')

            if current_stack_selections and meta_learner:                
                time.sleep(0.2)

                st.markdown(f"Training Stack {i+1} with models: {', '.join(current_stack_selections)}")
                trained_model = train_stack_model(current_stack_selections, meta_learner, x_train_resampled, y_train_resampled)
                st.session_state.trained_stacks[f"stack_model_{i+1}"] = trained_model

                st.success(f"✅ Stack{i + 1} trained")   
                progress_bar.progress((i + 1) / num_stacks)
            else:
                st.warning(f"No models selected for Stack {i+1}. Please Input the Models")
                break
        
        stack_models = st.session_state.trained_stacks
        if len(stack_models) > 2:
            st.subheader("Training Final Voting Classifier...")
            stack_models = list(stack_models.item())

            voting_ensemble = VotingClassifier(estimators=stack_models, voting=vote_strategy.lower(), n_jobs=-1)
            voting_ensemble.fit(x_train_resampled, y_train_resampled)
            y_pred = voting_ensemble.predict(x_test)

            model_download = voting_ensemble
        else:
            y_pred = stack_models["stack_model_1"].predict(x_test)
            model_download = stack_models["stack_model_1"]

        col1, col2 = st.columns(2)
        with col1 :
            # Confusion Matrix
            st.markdown("#### 📊 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            # Classification Report
            st.markdown("#### 📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            acc = accuracy_score(y_test, y_pred) * 100
            
            # Metrics
            st.markdown(
                f"""
                <div class='metric-box'>
                    <h4>Overall Accuracy</h4>
                    <div class='value'>{acc:.2f} %</div>
                </div>
                """
            , unsafe_allow_html=True)
            st.markdown('')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F1-Score (Hoax)", f"{report['1']['f1-score'] * 100:.2f}%", delta="10%", border=True)
            with col2:
                st.metric("Precision (Hoax)", f"{report['1']['precision'] * 100:.2f}%", delta="10%", border=True)
            with col3:
                st.metric("Recall (Hoax)", f"{report['1']['recall'] * 100:.2f}%", delta="10%", border=True)

            download_model(model_download)

        st.markdown("#### 📊 Evaluation True & Prediction")
        data = pd.DataFrame(
            {
                "Text" : x_text_test,
                "Prediction Label" : y_pred,
                "True Label" : y_test,
            }
        )
        st.dataframe(data, use_container_width=True)

