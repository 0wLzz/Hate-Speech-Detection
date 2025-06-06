import streamlit as st

def home_page():
    st.markdown('<h1 class="header-text">🕵️ Hoax Detection Application</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:735/1*vNwJyRGHG-MyeYcCnuzXew.png", 
                width=500, 
                caption="Fake News Meme",
                use_container_width=True)
    
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>👋 Welcome to Our Hoax Detection Platform!</h3>
            <p style='font-size: 1.2rem;'>
                This application is designed to help detect <strong style='color: #4b6cb7;'> HOAXES </strong> in Indonesian social media text using advanced <strong style='color: #4b6cb7;'>Stacking & Voting Ensemble Machine Learning </strong> techniques that enhance classification accuracy through model combination.
                <br>
                <br>
                The project was developed as part of our Final Assignment for the Machine Learning course, under the supervision of Pak Johannes by:
            </p>
            <ul style='font-size: 1.2rem;'>
                <li>Brian Juniarta Darmadi</li>
                <li>Nur Farhayati</li>
                <li>Owen Limantoro</li>
            </ul>
            <p style='font-size: 1.2rem;'>
                The driving motivation for this project is to contribute in the prevention of misinformation or disinformation spreading from the internet and especially in family groups. Which can mislead the public, cause confusion, divide communities, and in extreme cases, pose risks to the safety of the . In today’s digital era, anyone can publish and share information instantly, making it increasingly difficult to distinguish between false content.
                <br>
                <br>
                Therefore, we aimed to build a technology-driven tool that can automatically and efficiently verify the authenticity of social media content. By leveraging ensemble learning—specifically stacking and voting strategies—we combine the strengths of multiple machine learning models to produce more accurate and robust predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <div class="card">
        <h3 style='color: #2c3e50; margin-bottom: 1rem;'>🔍 Explore Our Features</h3>
        <ul style='margin-right:1rem; font-size: 1.2rem;'>
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
    
    st.markdown("<h3 style='color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; text-align:center;'>Navigate to Key Sections</h3>", unsafe_allow_html=True)
    
    home_feature_cards = [
        {"icon": "📊", "title": "EDA", "desc": "Explore the dataset with interactive visualizations", "target_page": "EDA"},
        {"icon": "🧩", "title": "Stacking & Voting", "desc": "Learn about ensemble techniques", "target_page": "Stacking & Voting Model"},
        {"icon": "🤖", "title": "Model Training", "desc": "Train and evaluate ML models", "target_page": "Model Training & Evaluation"},
        {"icon": "📈", "title": "Performance", "desc": "Compare model metrics", "target_page": "Model Training & Evaluation"},
        {"icon": "🔍", "title": "Hoax Detection", "desc": "Try our trained collection of models", "target_page": "Hoax Detector"}
    ]
    
    card_cols = st.columns(len(home_feature_cards))
    
    for i, card_item in enumerate(home_feature_cards):
        with card_cols[i]:
            st.markdown(f"""
            <div class="feature-card" style="padding: 15px; margin-bottom: 10px; text-align: center;">
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{card_item['icon']}</div>
                <h4 style='margin: 0.5rem 0; color: #2c3e50;'>{card_item['title']}</h4>
                <p style='font-size: 0.85rem; color: #7f8c8d; min-height: 70px; margin-top: 0.5rem; margin-bottom:1rem;'>{card_item['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Go to {card_item['title']}", key=f"home_card_btn_{card_item['title']}", use_container_width=True):
                st.session_state.active_page = card_item['target_page']
                st.rerun()
    
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