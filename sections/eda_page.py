import streamlit as st

def eda_page(raw_dataset, clean_dataset):
    st.markdown('<h1 class="header-text">📊 Dataset Explanation & EDA</h1>', unsafe_allow_html=True)
    st.markdown('')
    
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