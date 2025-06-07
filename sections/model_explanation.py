import streamlit as st

def model_explanation_page():
# Stacking & Voting Explanation Page
    st.markdown('<h1 class="header-text">⚙️ Stacking & Voting Explanation</h1>', unsafe_allow_html=True)
    st.markdown('')

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
            <h3 style='color: #2c3e50;'>How Voting Works</h3>
            <div style='text-align: center; margin: 1rem 0;'>
                <img src='https://miro.medium.com/v2/resize:fit:1400/1*djKLooxyOLvr98oMi5uwgA.jpeg' style='max-width: 100%; border-radius: 8px; ' alt='Stacking diagram'>
            </div>
            <p style='text-align: center; font-size: 0.85rem; color: #7f8c8d;'>
            Diagram showing the voting ensemble workflow with base models and voting output
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Proposed Method</h3>
            <p>
                In the proposed method, we implemented a combination of ensemble learning techniques: 
                <strong>Stacking</strong> and <strong>Voting</strong>. Stacking involves training multiple 
                base learners and combining their predictions through a meta-learner. This approach 
                aims to improve generalization performance by leveraging the strengths of different models. 
                Meanwhile, the Voting mechanism helps aggregate predictions either by majority (for classification) 
                or averaging (for regression) from multiple models to enhance robustness.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="card">
                <h3 style='color: #2c3e50;'>How Voting Works</h3>
            </div>
            """, unsafe_allow_html=True)

        st.image("Stacking + Voting.jpg", use_container_width =True, caption="Diagram showing the voting ensemble workflow with base models and voting output")

        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Stacking Combinations Used</h3>
            <p>
                Below are the stacking combinations used in our model training process. All stacking configurations used 
                <strong>Logistic Regression</strong> as the meta-learner.
            </p>
            <table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
                <thead>
                    <tr style="background-color: #ecf0f1; text-align: left;">
                        <th style="padding: 8px; border: 1px solid #bdc3c7;">Stack</th>
                        <th style="padding: 8px; border: 1px solid #bdc3c7;">Base Learners</th>
                        <th style="padding: 8px; border: 1px solid #bdc3c7;">Meta-Learner</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">1</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Random Forest, KNN, Decision Tree</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">2</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">XGBoost, Extra Trees, Random Forest</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">3</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">BernoulliNB, Ridge, Random Forest</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">4</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression, Ridge, PassiveAggressive</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">5</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">KNN, Decision Tree, BernoulliNB</td>
                        <td style="padding: 8px; border: 1px solid #bdc3c7;">Logistic Regression</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
