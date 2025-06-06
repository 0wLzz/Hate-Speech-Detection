import streamlit as st

def detector_page(vectorizer):
    st.markdown('<h1 class="header-text">🕵️ Hoax Detector</h1>', unsafe_allow_html=True)
    st.markdown('')


    # --- 1. Model Selection ---
    st.markdown("""
    <div class="card">
        <h3 style='color: #2c3e50;'>Select a Trained Model</h3>
        <p>Choose one of the ensemble models you trained in this session to make a prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    

    col_select, col_train = st.columns([5, 1])

    with col_train:
        st.markdown('Want to add a new model🤔⁉️')
        st.markdown('Make it custom down here ⤵️', unsafe_allow_html=True)
        if st.button("👉 Go to Model Training 👈"):
            st.session_state.active_page = "Model Training & Evaluation"
            st.rerun()

    with col_select:
        available_models = list(st.session_state.trained_stacks.keys())
        selected_model_name = st.selectbox(
            label="Select your trained model:",
            options=available_models
        )
    st.markdown("---")


    # --- 2. User Input ---
    st.markdown("""
    <div class="card">
        <h3 style='color: #2c3e50;'>Enter Text to Analyze</h3>
        <p>Paste the Indonesian text you want to check for potential hoax content below.</p>
    </div>
    """, unsafe_allow_html=True)

    input_text = st.text_area(
        label="Input Text:",
        placeholder="Ketik atau tempel teks berita di sini...",
        height=150
    )


    # --- 3. Prediction Button ---
    if st.button("Analyze Text", use_container_width=True, type="primary"):
        if not input_text:
            st.warning("Please enter some text to analyze.", icon="⚠️")
        elif not selected_model_name:
            st.warning("Something went wrong, no model appears to be selected.", icon="🤖")
        else:
            # --- 4. Prediction Logic ---
            # Retrieve the selected model object
            model_to_use = st.session_state.trained_stacks[selected_model_name]

            # **CRITICAL STEP**: Preprocess the input text using the SAME vectorizer
            # The .transform() method expects a list or iterable of documents
            processed_input = vectorizer.transform([input_text])

            # Make predictions
            with st.spinner("Analyzing..."):
                prediction = model_to_use.predict(processed_input)[0] # Get the first (and only) prediction
                
                # Get probability scores if the model supports it
                try:
                    probabilities = model_to_use.predict_proba(processed_input)[0]
                    confidence = probabilities[prediction]
                except AttributeError:
                    confidence = None

            # --- 5. Display Results ---
            st.markdown("---")
            st.subheader("🕵️‍♀️ Analysis Result")
            
            if prediction == 1: # Hoax
                st.error("Prediction: **HOAX DETECTED**", icon="❌")
                st.markdown("Our analysis suggests this text has characteristics commonly found in hoax or false information.")
                if confidence is not None:
                    st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                    st.progress(confidence, text=f"{confidence:.0%} confident it is a Hoax")
            else: # Not a Hoax
                st.success("Prediction: **LIKELY NOT A HOAX**", icon="✅")
                st.markdown("Our analysis suggests this text appears to be legitimate.")
                if confidence is not None:
                    st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                    st.progress(confidence, text=f"{confidence:.0%} confident it is Not a Hoax")
            
            with st.expander("Why this prediction?"):
                st.info("This prediction is based on the patterns the machine learning model learned from thousands of news articles. It analyzes word choice, sentence structure, and other text features against known examples of hoaxes and legitimate news. A high confidence score means the model is very certain about its classification based on the data it was trained on.")





        