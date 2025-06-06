import streamlit as st
import pandas as pd

def eda_page(raw_dataset, clean_dataset):
    st.markdown('<h1 class="header-text">📊 Dataset Explanation & EDA</h1>', unsafe_allow_html=True)
    st.markdown('')
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1: 
            st.markdown("""
            <div class="card">
                <h3 style='color: #2c3e50;'> 📊 Distribution of the Dataset</h3>
            </div>
            """, unsafe_allow_html=True)
            label_dist = pd.DataFrame({
                "Label": ["Hoax", "Non-hoax"],
                "Count": [3465, 766]
            }).set_index("Label")

            st.bar_chart(label_dist, horizontal=True, height=470, width=100)

        with col2:
            st.markdown("""
            <div style="
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                font-family: 'Segoe UI', sans-serif;
            ">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">📄 Data Explanation</h3>
                <p style="color: #333; font-size: 15px; line-height: 1.6;">
                    This dataset, sourced from <strong>Hugging Face Datasets</strong>, comprises a collection of <strong>4,209 news articles</strong> in Bahasa Indonesia. Each article has been labeled as either <span style="color: red;"><strong>"Fake" (Hoax)</strong></span> or <span style="color: green;"><strong>"Non Fake" (Real News)</strong></span>.
                </p>
                <p style="color: #333; font-size: 15px; line-height: 1.6;">
                    A key characteristic of this dataset is its <strong>imbalanced class distribution</strong>, which reflects a real-world challenge for classification models:
                </p>
                <ul style="margin-left: 20px; color: #333; font-size: 15px;">
                    <li><strong>3,465</strong> Fake News instances</li>
                    <li><strong>766</strong> Real News instances</li>
                </ul>
                <p style="color: #333; font-size: 15px; line-height: 1.6;">
                    The original dataset includes the following columns:
                </p>
                <ul style="margin-left: 20px; color: #333; font-size: 15px;">
                    <li><strong>ID</strong>: A unique identifier for each news article.</li>
                    <li><strong>Tanggal</strong>: The date associated with the news article.</li>
                    <li><strong>Judul</strong>: The title of the news article.</li>
                    <li><strong>Narasi</strong>: The full text or narrative content.</li>
                    <li><strong>Nama File Gambar</strong>: Filename of any associated image.</li>
                </ul>
                <h3 style='color: #2c3e50; margin-top: 1rem;'>Dataset Source:</h3>
                <p>This app uses a dataset from <a href="https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews">Kaggle</a></p> 
            </div>
            """, unsafe_allow_html=True)

    # Toggle between raw and clean datasets
    option_map = { 0: "Raw", 1: "Clean"}
    selection = st.segmented_control(
        "Choose to see the dataset",
        options=option_map.values(),
        format_func=lambda option: option + " Dataset",
        selection_mode="single",
    )

    # Display dataset based on selection
    dataset_container = st.container()
    with dataset_container:
        if selection == "Raw":
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
        <p style="color: #666; margin-top: 0; font-size:1.2rem">
            Text Preprocessing is one of the initial steps of Natural Language Processing (NLP) that involves cleaning and transforming raw data into suitable data for further processing. It enhances the quality of the text makes it easier to work and improves the performance of machine learning models. Explore the text cleaning and processing steps applied to the raw data, so that the model can receive it
        </p>
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
    
    st.markdown("""
    <div class="card" style="padding-top:0.2rem">
        <h3 style="color: #2c3e50; margin-top: 1.5rem;">⚖️ Handling Imbalanced Dataset with SMOTE</h3>
        <p style="color: #666; font-size: 1rem; line-height: 1.6;">
            The dataset is highly imbalanced, with many more <strong>hoax</strong> than <strong>non-hoax</strong> examples. This imbalance can bias the model toward the majority class, leading to poor performance on underrepresented cases.
            To address this, we use <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong>, which:
        </p>
        <ul style="color: #444; padding-left: 1.2rem; font-size: 0.95rem;">
            <li>Generates new synthetic samples for the minority class (non-hoax)</li>
            <li>Prevents the model from learning a biased decision boundary</li>
            <li>Improves recall and F1-score for minority classes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📈 6. SMOTE"):        
        st.markdown("**Implementation:**")
        st.code(""" 
                from imblearn.over_sampling import SMOTE
                # oversampling the train dataset using SMOTE
                smt = SMOTE()
                X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
                """, language='python')