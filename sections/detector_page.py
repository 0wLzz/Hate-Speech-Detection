import streamlit as st

def detector_page(vectorizer):
    st.markdown('<h1 class="header-text">🕵️ Hoax Detector</h1>', unsafe_allow_html=True)
    st.markdown('')

    # Check if any models have been trained and stored in session state
    if not st.session_state.get("trained_stacks"):
        st.warning("⚠️ No models have been trained in this session yet. Please go to the 'Model Training & Evaluation' page to train a model first.", icon="🤖")

        if st.button("Go to Model Training"):
            st.session_state.active_page = "Model Training & Evaluation"
            st.rerun()
    else:
        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Select a Trained Model</h3>
            <p>Choose one of the ensemble models you trained in this session to make a prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        available_models = list(st.session_state.trained_stacks.keys())
        selected_model_name = st.selectbox(
            label="Select your trained model:",
            options=available_models
        )
        st.markdown("---")

        st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50;'>Enter Text to Analyze</h3>
            <p>Paste the Indonesian text you want to check for potential hoax content below.</p>
        </div>
        """, unsafe_allow_html=True)

        input_text = st.text_area(
            label="Input Text:",
            placeholder="Write or Paste your news here...",
            height=150
        )

        if st.button("Analyze Text", use_container_width=True, type="primary"):
            if not input_text:
                st.warning("Please enter some text to analyze.", icon="⚠️")
            elif not selected_model_name:
                st.warning("Something went wrong, no model appears to be selected.", icon="🤖")
            else:
                model_to_use = st.session_state.trained_stacks[selected_model_name]
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