# ==============================================================================
#  Real-Estate Data Analysis and LLM Application
# ==============================================================================
# This script performs the following tasks:
# 1. Sets up the environment by checking and installing necessary packages.
# 2. Downloads required open-source models for embeddings and language generation.
# 3. Provides a Streamlit UI with multiple tabs:
#    - Home: Welcome page.
#    - Data Exploration: Cleans, explores, and visualizes the real-estate data.
#    - Price Prediction: Trains a simple regression model and allows for predictions.
#    - Listing Q&A (RAG): Implements a RAG system to answer questions about listings.
#
#  Constraints Adhered To:
#  - Uses local/open-source models (TinyLlama for generation, all-MiniLM for embeddings).
#  - No paid API keys required.
#  - Avoids FAISS, using NumPy/SciPy for vector search.
#  - Avoids the 'sentence_transformers' library directly, using 'transformers' instead.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import importlib.util
import time

# --- 1. SETUP AND PREREQUISITE INSTALLATION ---

# Using a try-finally block to ensure messages are always displayed.
# The main logic is within the try block. The finally block is a good practice
# for cleanup or final status messages, though here it's for demonstration.
def check_and_install_packages():
    """
    Checks if required packages are installed. If not, it attempts to install them.
    This function is robust and provides clear feedback to the user.
    """
    st.header("️ Environment Setup")
    status_placeholder = st.empty()
    log_placeholder = st.expander("Installation Logs", expanded=False)
    
    required_packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "streamlit": "streamlit",
        "llama-cpp-python": "llama_cpp",
        "huggingface-hub": "huggingface_hub",
        "torch": "torch",
        "transformers": "transformers"
    }
    
    all_installed = True
    try:
        with st.spinner("Checking required packages..."):
            for package, import_name in required_packages.items():
                spec = importlib.util.find_spec(import_name)
                if spec is None:
                    all_installed = False
                    log_placeholder.info(f"Package '{package}' not found. Attempting to install...")
                    try:
                        # Using subprocess to install the package
                        process = subprocess.Popen(
                            [sys.executable, "-m", "pip", "install", package],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        # Stream output to the log expander
                        for line in iter(process.stdout.readline, ''):
                            log_placeholder.text(line.strip())
                        process.wait() # Wait for the process to complete
                        
                        # Cross-check installation
                        spec_after_install = importlib.util.find_spec(import_name)
                        if spec_after_install:
                            log_placeholder.success(f"Successfully installed '{package}'.")
                        else:
                            # This is a critical error
                            stderr_output = process.stderr.read()
                            log_placeholder.error(f"Failed to install '{package}'. Pip stderr: {stderr_output}")
                            st.error(f"Could not install '{package}'. Please install it manually (`pip install {package}`) and restart the app.")
                            st.stop() # Stop the app if a critical dependency fails
                            
                    except Exception as e:
                        log_placeholder.error(f"An error occurred during installation of '{package}': {e}")
                        st.error(f"Failed to install '{package}'. Please do so manually.")
                        st.stop()
    finally:
        # This block will execute whether the try block succeeded or failed.
        if all_installed:
            status_placeholder.success("All required packages are installed and ready!")
            time.sleep(2) # Give user time to read the message
            status_placeholder.empty() # Clear the success message
        else:
            status_placeholder.success("Package installation process finished. Please check logs for details.")
            st.info("Rerunning the app to load new packages...")
            st.rerun() # Rerun the script to import the newly installed packages

@st.cache_resource
def download_models():
    """
    Downloads the LLM and embedding models from Hugging Face Hub.
    Uses caching to prevent re-downloading on every app run.
    """
    st.header("Downloading AI Models")
    
    # --- LLM Model (GGUF format for llama-cpp-python) ---
    llm_model_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    llm_model_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    models_dir = "models"
    llm_model_path = os.path.join(models_dir, llm_model_filename)

    if not os.path.exists(llm_model_path):
        st.info(f"Downloading LLM: {llm_model_filename}. This may take a few minutes...")
        os.makedirs(models_dir, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            with st.spinner("Downloading... please wait."):
                hf_hub_download(
                    repo_id=llm_model_repo,
                    filename=llm_model_filename,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
            st.success("LLM model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download LLM. Error: {e}")
            st.stop()
    else:
        st.info("LLM model already exists locally.")

    # --- Embedding Model (Handled by transformers library) ---
    # We don't need to manually download it. Transformers will cache it on first use.
    # We just inform the user.
    st.info("Embedding model ('all-MiniLM-L6-v2') will be downloaded on first use by the 'transformers' library.")
    
    return llm_model_path

# --- 2. CORE APPLICATION LOGIC ---

# --- Data Loading and Caching ---
@st.cache_data
def load_and_clean_data(filepath):
    """
    Loads the dataset from the specified path, performs cleaning, and
    engineers a 'price_per_sqft' feature.
    """
    try:
        df = pd.read_csv(filepath)
        # Basic cleaning
        df.dropna(subset=['price', 'sqft', 'beds', 'baths', 'remarks'], inplace=True)
        df = df[df['sqft'] > 100] # Remove unrealistic data
        df = df[df['price'] > 10000]
        df['price_per_sqft'] = df['price'] / df['sqft']
        df['remarks'] = df['remarks'].astype(str)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{filepath}'.")
        st.error("Please make sure the 'Data' folder exists and contains 'listings_sample.csv'.")
        return None

# --- Predictive Model Training ---
@st.cache_resource
def train_prediction_model(df):
    """
    Trains a simple RandomForestRegressor to predict price.
    Caches the trained model and preprocessor.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    features = ['beds', 'baths', 'sqft', 'city']
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a preprocessor for categorical features
    categorical_features = ['city']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', one_hot_encoder, categorical_features)],
        remainder='passthrough'
    )
    
    # Create the pipeline with the preprocessor and the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2, df['city'].unique().tolist()

# --- RAG System Components ---
@st.cache_resource
def load_embedding_model():
    """Loads the sentence embedding model from Hugging Face."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Could not load embedding model. Error: {e}")
        return None, None

def get_embeddings(texts, _tokenizer, _model):
    """Generates embeddings for a list of texts."""
    import torch
    # Mean Pooling function
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = _tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = _model(**encoded_input)
    
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.numpy()

@st.cache_data
def generate_corpus_embeddings(_df, _tokenizer, _model):
    """Generates and caches embeddings for all listing remarks."""
    with st.spinner("Creating embeddings for all listings... This is a one-time process."):
        # corpus_embeddings = get_embeddings(_df['remarks'].tolist(), _tokenizer, _model)
        
        # Get corpus embeddings based on get_listing_structured_info function
        structured_texts = _df.apply(get_listing_structured_info, axis=1).tolist()
        corpus_embeddings = get_embeddings(structured_texts, _tokenizer, _model)

    return corpus_embeddings

@st.cache_resource
def load_llm(_model_path):
    """Loads the GGUF language model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=_model_path,
            n_ctx=2048,      # Context window size
            n_gpu_layers=-1, # Offload all possible layers to GPU
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM from path '{_model_path}'. Error: {e}")
        st.warning("The RAG feature will not work. Please ensure you have a compatible system (C++ compiler) and the model file is correct.")
        return None

def find_top_k_similar(query_embedding, corpus_embeddings, k=3):
    """
    Finds the top-k most similar items in the corpus using cosine similarity.
    This is the alternative to FAISS.
    """
    from scipy.spatial.distance import cdist
    # cdist computes distance, so 1 - distance gives similarity
    similarities = 1 - cdist(query_embedding, corpus_embeddings, 'cosine')[0]
    # Get the indices of the top-k similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices

def llama_generate_summary_prompt(row):
    """Creates a prompt for the LLM to generate a listing summary."""
    # Note: The f-string formatting is corrected from the user prompt's example.
    # Added formatting for price and safe .get() for all fields.
    price_str = f"${int(row.get('price', 0)):,}" if pd.notna(row.get('price')) else "N/A"
    
    prompt = (
        "Write a 2-3 sentence listing summary including address, price, beds, baths and a short supporting remark.\n\n"
        f"Address: {row.get('address', 'N/A')}\n"
        f"Price: {price_str}\n"
        f"Beds: {row.get('beds', 'N/A')}\n"
        f"Baths: {row.get('baths', 'N/A')}\n"
        f"Remarks: {row.get('remarks', 'N/A')}\n\n"
        "Summary:"
    )
    return prompt


def get_listing_summary(llm, row):
    """Generates a summary for a listing using the LLM."""
    prompt_text = llama_generate_summary_prompt(row)
    
    prompt_template = f"""
    <|system|>
    You are a helpful real estate assistant. Your task is to generate a concise 2-3 sentence summary for a property listing based on the provided details.</s>
    <|user|>
    {prompt_text}</s>
    <|assistant|>
    """
    
    try:
        output = llm(prompt_template, max_tokens=150, stop=["<|user|>"], echo=False)
        summary = output['choices'][0]['text'].strip()
        return summary
    except Exception as e:
        st.warning(f"Could not generate summary for listing {row.get('id', 'N/A')}. Error: {e}")
        return "Could not generate summary."
    
def get_listing_structured_info(row):
    """Formats the listing information for display."""
    price_str = f"${int(row.get('price', 0)):,}" if pd.notna(row.get('price')) else "N/A"
    
    info = (
        f"Remarks: {row.get('remarks', 'N/A')}\n\n"
        f"ID: `{row.get('id', 'N/A')}`\n"
        f"Address: {row.get('address', 'N/A')}\n"
        f"Price: {price_str}\n"
        f"Sqft: {row.get('sqft', 'N/A')}\n"
        f"Price per SqFt: ${row.get('price_per_sqft', 'N/A'):.2f}" if pd.notna(row.get('price_per_sqft')) else "Price per SqFt: N/A"
        f"City: {row.get('city', 'N/A')}\n"
        f"State: {row.get('state', 'N/A')}\n"
        f"Beds: {row.get('beds', 'N/A')}\n"
        f"Baths: {row.get('baths', 'N/A')}\n"
        f"List date: {row.get('list_date', 'N/A')}\n"
        f"Agent ID: `{row.get('agent_id', 'N/A')}`"
    )
    return info
# --- 3. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Real Estate AI Assistant", layout="wide", initial_sidebar_state="expanded")

    # --- Initial Setup Checks ---
    with st.sidebar:
        st.title("🏠 Real Estate AI")
        st.info("This app analyzes real estate data, predicts prices, and answers questions about listings.")
    
    # Run setup checks only once at the beginning
    if 'setup_done' not in st.session_state:
        check_and_install_packages()
        download_models()
        st.session_state['setup_done'] = True
        st.rerun()

    # --- Load Data ---
    data_path = os.path.join("Data", "listings_sample.csv")
    df = load_and_clean_data(data_path)

    if df is None:
        st.stop() # Stop execution if data loading failed

    # --- Sidebar Navigation ---
    with st.sidebar:
        page = st.radio(
            "Choose a feature",
            ("Data Exploration", "Price Prediction", "Listing Q&A (RAG)"),
            captions=["Analyze the data", "Estimate property values", "Ask about listings"]
        )
        st.markdown("---")
        st.markdown(
            "**Models Used:**\n"
            "- **Prediction:** `RandomForestRegressor` (Scikit-learn)\n"
            "- **Embeddings:** `all-MiniLM-L6-v2` (Hugging Face)\n"
            "- **Q&A:** `TinyLlama-1.1B-Chat` (GGUF)"
        )

    # --- Page Content ---
    if page == "Data Exploration":
        render_data_exploration(df)
    elif page == "Price Prediction":
        render_price_prediction(df)
    elif page == "Listing Q&A (RAG)":
        render_rag_qa(df)

def render_data_exploration(df):
    st.title("📊 Data Exploration and Visualization")
    st.markdown("An overview of the real estate dataset. We've cleaned the data by removing rows with missing critical values.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    with st.expander("Show Dataset Statistics"):
        st.write(df.describe())

    st.subheader("Visual Comparisons")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Price by City")
        avg_price_city = df.groupby('city')['price'].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=avg_price_city.index, y=avg_price_city.values, ax=ax1, palette="viridis")
        ax1.set_ylabel("Average Price ($)")
        ax1.set_xlabel("City")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Price Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax2, color="skyblue")
        ax2.set_xlabel("Price ($)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
        
    st.markdown("#### Price per Square Foot vs. Square Footage")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='sqft', y='price_per_sqft', hue='city', alpha=0.6, ax=ax3)
    ax3.set_xlabel("Square Footage (sqft)")
    ax3.set_ylabel("Price per SqFt ($)")
    st.pyplot(fig3)

def render_price_prediction(df):
    st.title("📈 Price Prediction Model")
    st.markdown("Predict the price of a listing based on its features. We use a `RandomForestRegressor` model for this task.")
    
    model, mae, r2, city_options = train_prediction_model(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model R² Score", value=f"{r2:.2f}")
    with col2:
        st.metric(label="Mean Absolute Error", value=f"${mae:,.0f}")
    st.info("The R² score indicates the model explains a good portion of the price variance. The MAE is the average prediction error in dollars.")

    st.subheader("Get a Price Estimate")
    with st.form("prediction_form"):
        beds = st.slider("Number of Bedrooms", 1, 10, 3)
        baths = st.slider("Number of Bathrooms", 1, 8, 2)
        sqft = st.number_input("Square Footage (sqft)", min_value=300, max_value=10000, value=1500, step=50)
        city = st.selectbox("City", options=city_options)
        
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_data = pd.DataFrame([[beds, baths, sqft, city]], columns=['beds', 'baths', 'sqft', 'city'])
        with st.spinner("Calculating..."):
            prediction = model.predict(input_data)[0]
        st.success(f"**Predicted Price: ${prediction:,.2f}**")

def render_rag_qa(df):
    st.title("💬 Listing Q&A (RAG)")
    st.markdown("Ask a question about the property listings, and the AI will find relevant information and provide an answer.")
    st.warning("This AI answers **only** based on the 'remarks' column of the listings. It may not be perfect and works best with specific questions (e.g., 'which house has a new roof?').", icon="⚠️")

    # --- Load models for this page ---
    llm_model_path = os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    llm = load_llm(llm_model_path)
    tokenizer, embedding_model = load_embedding_model()

    if not all([llm, tokenizer, embedding_model]):
        st.error("One or more AI models failed to load. This feature is unavailable.")
        return

    corpus_embeddings = generate_corpus_embeddings(df, tokenizer, embedding_model)

    user_question = st.text_input("Ask your question about the listings:", placeholder="e.g., Are there any properties with a fenced yard?")

    if user_question:
        with st.spinner("Searching for relevant listings and generating an answer..."):
            # 1. Embed the user's query
            query_embedding = get_embeddings([user_question], tokenizer, embedding_model)
            
            # 2. Find top-k similar remarks (the "Retrieval" part)
            top_k = 3
            retrieved_indices = find_top_k_similar(query_embedding, corpus_embeddings, k=top_k)
            # retrieved_remarks = df.iloc[retrieved_indices]['remarks'].tolist()
            retrieved_remarks = df.iloc[retrieved_indices].apply(get_listing_structured_info, axis=1).tolist()  
                        
            # 3. Construct the prompt for the LLM (the "Generation" part)
            context = "\n\n".join([f"Listing {i+1}: {remark}" for i, remark in enumerate(retrieved_remarks)])
            
            prompt_template = f"""
            <|system|>
            You are a helpful real estate assistant. Your task is to answer the user's question based *only* on the context provided below.
            If the context does not contain the answer, state that you cannot find the information in the provided listings.
            Do not make up information. Be concise and directly answer the question.</s>
            <|user|>
            CONTEXT:
            ---
            {context}
            ---
            QUESTION: {user_question}</s>
            <|assistant|>
            """
            
            # 4. Get the LLM's response
            try:
                # output = llm(prompt_template, max_tokens=250, stop=["<|user|>"], echo=False)
                output = llm(prompt_template, max_tokens=500, stop=["<|user|>"], echo=False)
                answer = output['choices'][0]['text'].strip()
                
                st.subheader("Answer:")
                st.success(answer)

                # To handle the "incorrect answers" problem, we show the user the exact context the LLM used.
                with st.expander("Show Retrieved Context (What the AI used to answer)"):
                    st.info("The AI's answer is based on the following property listings. A summary for each has been generated.")
                    with st.spinner("Generating summaries for retrieved listings..."):
                        for i, idx in enumerate(retrieved_indices):
                            listing_row = df.iloc[idx]
                            
                            # --- AUTOMATIC SUMMARY GENERATION ---
                            summary = "Could not generate summary." # Default value
                            try:
                                # 1. Create the summary prompt
                                summary_prompt_text = llama_generate_summary_prompt(listing_row)
                                
                                # 2. Format it for the chat model
                                summary_prompt_template = f"""
                                <|system|>
                                You are a helpful real estate assistant. Your task is to generate a concise 2-3 sentence summary for a property listing based on the provided details.</s>
                                <|user|>
                                {summary_prompt_text}</s>
                                <|assistant|>
                                """
                                
                                # 3. Call the LLM
                                summary_output = llm(summary_prompt_template, max_tokens=150, stop=["<|user|>"], echo=False)
                                summary = summary_output['choices'][0]['text'].strip()
                            except Exception as e:
                                # Log the error but don't stop the app
                                st.warning(f"Could not generate summary for listing {listing_row['id']}. Error: {e}")

                            # --- DISPLAY STRUCTURED INFO ---
                            st.markdown(f"---")
                            st.markdown(f"#### Source {i+1}: Listing from {listing_row['city']}")
                            
                            # Format price for better display
                            price_str = f"${int(listing_row.get('price', 0)):,}" if pd.notna(listing_row.get('price')) else "N/A"

                            # Using a more structured layout
                            st.markdown(f"""
                            - **ID:** `{listing_row['id']}`
                            - **Address:** {listing_row.get('address', 'N/A')}
                            - **Price:** {price_str}
                            - **Beds:** {listing_row.get('beds', 'N/A')} | **Baths:** {listing_row.get('baths', 'N/A')}
                            """)
                            st.markdown(f"**Remarks:**")
                            st.info(f"{listing_row['remarks']}")
                            st.markdown(f"**Generated Summary:**")
                            st.success(f"{summary}")

            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")


if __name__ == "__main__":
    main()
