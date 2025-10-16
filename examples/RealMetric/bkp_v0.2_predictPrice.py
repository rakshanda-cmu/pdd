# ==============================================================================
# 1. SETUP AND PREREQUISITE INSTALLATION
# ==============================================================================
import sys
import subprocess
import os
import pkg_resources
from pathlib import Path

# --- Define required packages ---
# We separate llama-cpp-python because it needs special handling
REQUIRED_PACKAGES = {
    "streamlit": "1.34.0",
    "pandas": "2.2.2",
    "scikit-learn": "1.4.2",
    "plotly_express": "0.4.1",
    "numpy": "1.26.4",
    "torch": "2.3.0",
    "transformers": "4.40.1",
    "huggingface_hub": "0.34.0",
}
LLAMA_CPP_PACKAGE = "llama-cpp-python"
LLAMA_CPP_VERSION = "0.2.75"

def install_and_check_packages():
    """
    Checks for required packages and installs them if missing.
    Handles the special case of llama-cpp-python.
    """
    print("--- Checking and installing prerequisites ---")
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg for pkg in REQUIRED_PACKAGES if pkg not in installed_packages]
    
    # --- Standard packages installation ---
    if missing_packages:
        print(f"Missing standard packages: {missing_packages}. Installing...")
        try:
            # Using list comprehension to create the package==version string
            pip_args = [sys.executable, '-m', 'pip', 'install'] + [f"{pkg}=={ver}" for pkg, ver in REQUIRED_PACKAGES.items() if pkg in missing_packages]
            subprocess.check_call(pip_args)
            print("Standard packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install standard packages: {e}")
            print("Please try installing them manually using pip.")
            sys.exit(1)

    # --- Special handling for llama-cpp-python ---
    llama_cpp_installed = False
    try:
        # A more reliable check than just pkg_resources
        import llama_cpp
        llama_cpp_installed = True
        print(f"'{LLAMA_CPP_PACKAGE}' is already installed.")
    except ImportError:
        print(f"'{LLAMA_CPP_PACKAGE}' not found. Attempting installation...")
        # This is a common fix for compilation issues on many systems
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=ON" # Example for NVIDIA GPUs, adjust if needed
        env["FORCE_CMAKE"] = "1"
        
        try:
            # Using a try...except...finally block as requested
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', f"{LLAMA_CPP_PACKAGE}=={LLAMA_CPP_VERSION}"],
                env=env
            )
            print(f"'{LLAMA_CPP_PACKAGE}' installed successfully.")
            llama_cpp_installed = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install '{LLAMA_CPP_PACKAGE}'.")
            print("This package often requires a C++ compiler (like build-essential on Linux or Visual Studio Build Tools on Windows).")
            print("Please visit the llama-cpp-python GitHub page for detailed installation instructions for your OS.")
            print(f"Error details: {e}")
        finally:
            # This 'finally' block ensures we always re-check and inform the user
            print("--- Final check for llama-cpp-python ---")
            try:
                import llama_cpp
                print("Final check successful. Llama-CPP is available.")
                llama_cpp_installed = True
            except ImportError:
                print("Final check failed. Llama-CPP is not available. The Q&A tab will be disabled.")
                llama_cpp_installed = False
                
    print("--- Prerequisite check complete ---")
    return llama_cpp_installed

# --- Run the installation ---
LLAMA_CPP_AVAILABLE = install_and_check_packages()

# ==============================================================================
# 2. IMPORT LIBRARIES (after installation)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
import torch

# Only import LlamaCpp if it was successfully installed
if LLAMA_CPP_AVAILABLE:
    from llama_cpp import Llama

# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================

def create_sample_data_if_not_exists():
    """Creates a dummy CSV file for the app if it doesn't exist."""
    data_dir = Path("Data")
    data_file = data_dir / "listings_sample.csv"

    if not data_file.is_file():
        print(f"'{data_file}' not found. Creating sample data.")
        data_dir.mkdir(exist_ok=True)
        
        sample_data = {
            'price': [250000, 750000, 450000, 1200000, 350000, 620000, 890000, 295000, 510000, 1500000],
            'beds': [2, 4, 3, 5, 3, 4, 4, 2, 3, 5],
            'baths': [1, 3, 2, 4, 2, 3, 3, 2, 3, 5],
            'sqft': [1100, 2500, 1800, 4000, 1600, 2200, 3100, 1300, 1950, 4500],
            'city': ['Springfield', 'Metropolis', 'Springfield', 'Metropolis', 'Gotham', 'Gotham', 'Metropolis', 'Springfield', 'Gotham', 'Metropolis'],
            'remarks': [
                'Cozy starter home with a large backyard. Recently updated kitchen with granite countertops. Close to parks and schools.',
                'Luxurious executive home in a gated community. Features a home theater, pool, and a three-car garage. Breathtaking city views.',
                'Charming bungalow with original hardwood floors. Fenced yard perfect for pets. Walking distance to downtown shops.',
                'Sprawling estate with a guest house and tennis court. Modern architecture and high-end finishes throughout. Ideal for entertaining.',
                'Well-maintained family home in a quiet suburb. New roof in 2021. Finished basement provides extra living space.',
                'Beautiful two-story colonial with a spacious master suite. The kitchen includes stainless steel appliances. Great school district.',
                'Stunning modern home with an open floor plan and smart home technology. Energy-efficient windows and solar panels included.',
                'Perfect condo for a young professional. Low maintenance living with access to a community gym and pool. Great location.',
                'Classic brick home with a large front porch. The property includes a detached workshop. Mature trees provide ample shade.',
                'Magnificent mansion on a private lot. Features a wine cellar, library, and panoramic windows. Unparalleled craftsmanship.'
            ]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(data_file, index=False)
        print("Sample data created successfully.")

# --- Create data file on first run ---
create_sample_data_if_not_exists()

@st.cache_data
def load_and_clean_data(filepath="Data/listings_sample.csv"):
    """Loads the real estate data, cleans it, and engineers features."""
    df = pd.read_csv(filepath)
    df.dropna(subset=['price', 'sqft', 'beds', 'baths'], inplace=True)
    df = df[df['sqft'] > 0] # Avoid division by zero
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['remarks'] = df['remarks'].astype(str)
    return df

# ==============================================================================
# 4. UI - MAIN APPLICATION LAYOUT
# ==============================================================================

st.set_page_config(page_title="Real Estate AI Assistant", layout="wide")
st.title("🏡 Real Estate Data Explorer & AI Assistant")

# --- Load data ---
df = load_and_clean_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    ["1. Data Exploration & Visualization", "2. Price Prediction Model", "3. Listing Q&A (RAG)"]
)

# ==============================================================================
# 5. UI - SECTION 1: DATA EXPLORATION
# ==============================================================================
if app_mode == "1. Data Exploration & Visualization":
    st.header("1. Data Exploration & Visualization")
    st.markdown("Here is a preview of the cleaned dataset. We've added a `price_per_sqft` column for better analysis.")
    st.dataframe(df)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Interactive Visualizations")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Price by City")
        avg_price_city = df.groupby('city')['price'].mean().sort_values(ascending=False)
        fig1 = px.bar(avg_price_city, x=avg_price_city.index, y='price', labels={'price': 'Average Price', 'x': 'City'}, color=avg_price_city.index)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Price vs. Square Footage")
        fig2 = px.scatter(df, x='sqft', y='price', color='city', hover_data=['beds', 'baths'], labels={'sqft': 'Square Feet', 'price': 'Price'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Distribution of Price Per Square Foot")
    fig3 = px.histogram(df, x='price_per_sqft', nbins=20, title="Distribution of Price/SqFt")
    st.plotly_chart(fig3, use_container_width=True)

# ==============================================================================
# 6. UI - SECTION 2: PREDICTIVE MODEL
# ==============================================================================
elif app_mode == "2. Price Prediction Model":
    st.header("2. Price Prediction Model")
    st.info("**Model Used:** A simple `Linear Regression` model from Scikit-learn.")
    st.markdown("""
    This model predicts the **Price per Square Foot** based on the number of bedrooms, bathrooms, and the total square footage. 
    You can input your own values below to see a prediction.
    """)

    # --- Model Training (cached to avoid retraining on every interaction) ---
    @st.cache_resource
    def train_price_model(data):
        features = ['beds', 'baths', 'sqft']
        target = 'price_per_sqft'
        X = data[features]
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mae, r2

    model, mae, r2 = train_price_model(df)

    st.subheader("Get a Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        beds_input = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    with col2:
        baths_input = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    with col3:
        sqft_input = st.number_input("Square Feet", min_value=500, max_value=10000, value=1800, step=100)

    if st.button("Predict Price"):
        input_data = pd.DataFrame([[beds_input, baths_input, sqft_input]], columns=['beds', 'baths', 'sqft'])
        predicted_price_per_sqft = model.predict(input_data)[0]
        predicted_total_price = predicted_price_per_sqft * sqft_input
        
        st.success(f"**Predicted Price per SqFt:** ${predicted_price_per_sqft:,.2f}")
        st.success(f"**Estimated Total Price:** ${predicted_total_price:,.2f}")

    st.subheader("Model Performance")
    st.markdown(f"The model was trained on 80% of the data and evaluated on the remaining 20%.")
    st.metric(label="R-squared (R²)", value=f"{r2:.2f}")
    st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}")
    st.caption("R² indicates how much of the variance in price is explained by the features. MAE is the average absolute difference between predicted and actual prices.")

# ==============================================================================
# 7. UI - SECTION 3: RAG Q&A
# ==============================================================================
elif app_mode == "3. Listing Q&A (RAG)":
    st.header("3. Listing Q&A (RAG)")

    if not LLAMA_CPP_AVAILABLE:
        st.error(
            "**LLM Functionality Disabled.**\n\n"
            "The `llama-cpp-python` package could not be installed or imported. "
            "Please check the console output for installation errors and follow the instructions to install it manually."
        )
    else:
        st.info("""
        **Models Used:**
        - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (run locally via `transformers`). This model turns text into numerical vectors.
        - **LLM:** `Phi-3-mini-4k-instruct-q4.gguf` (a small, powerful model run locally via `llama-cpp-python`). This model generates answers.
        
        This is a mini RAG (Retrieval-Augmented Generation) system. Ask a question about features in the listings (e.g., "Which homes have a pool?"), and the system will:
        1.  Find the most relevant listing remarks using vector similarity.
        2.  Feed those remarks to the local LLM as context.
        3.  Generate an answer based *only* on the provided context.
        """)

        # --- Helper Functions for RAG ---
        @st.cache_resource
        def get_embedding_model():
            """Loads the embedding model and tokenizer from Hugging Face."""
            with st.spinner("Loading embedding model... This may take a moment on first run."):
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            return tokenizer, model

        def mean_pooling(model_output, attention_mask):
            """Helper function for sentence embeddings."""
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        @st.cache_data
        def create_embeddings(_df, _tokenizer, _model):
            """Creates embeddings for all remarks in the dataframe."""
            remarks = _df['remarks'].tolist()
            with st.spinner("Creating text embeddings for all listings..."):
                encoded_input = _tokenizer(remarks, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    model_output = _model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.numpy()

        def find_similar_remarks(query_embedding, remark_embeddings, top_k=3):
            """
            Finds the top_k most similar remarks using cosine similarity.
            This is our alternative to Faiss.
            """
            # Cosine similarity is the dot product of normalized vectors
            similarities = np.dot(remark_embeddings, query_embedding.T).flatten()
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            return top_k_indices

        @st.cache_resource
        def get_llm():
            """Downloads and loads the GGUF model."""
            with st.spinner("Loading local LLM (Phi-3-mini)... This is a one-time download (~2.2 GB)."):
                model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"
                model_file = "Phi-3-mini-4k-instruct-q4.gguf"
                try:
                    model_path = hf_hub_download(repo_id=model_name, filename=model_file)
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=2048, # Context window
                        verbose=False
                    )
                    return llm
                except Exception as e:
                    st.error(f"Failed to load LLM: {e}")
                    return None

        # --- Main RAG Logic ---
        tokenizer, embedding_model = get_embedding_model()
        remark_embeddings = create_embeddings(df, tokenizer, embedding_model)
        llm = get_llm()

        if llm:
            user_question = st.text_input("Ask a question about the listings:", "Which homes have a three-car garage or a pool?")

            if st.button("Get Answer"):
                if user_question:
                    with st.spinner("Searching for relevant listings and generating an answer..."):
                        # 1. Embed the user's query
                        encoded_query = tokenizer([user_question], padding=True, truncation=True, return_tensors='pt')
                        with torch.no_grad():
                            query_output = embedding_model(**encoded_query)
                        query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
                        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).numpy()

                        # 2. Retrieve top-k relevant remarks
                        top_indices = find_similar_remarks(query_embedding, remark_embeddings, top_k=3)
                        context_remarks = df['remarks'].iloc[top_indices].tolist()
                        context = "\n\n".join(f"Listing {i+1}: {remark}" for i, remark in enumerate(context_remarks))

                        # 3. Build the prompt for the LLM
                        prompt = f"""
                        <|system|>
                        You are a helpful real estate assistant. Answer the user's question based *only* on the context provided below.
                        If the information is not in the context, say "I cannot find information about that in the provided listings."
                        Do not make up information. Be concise.
                        <|end|>
                        <|user|>
                        CONTEXT:
                        ---
                        {context}
                        ---
                        QUESTION: {user_question}
                        <|end|>
                        <|assistant|>
                        """

                        # 4. Generate the answer
                        output = llm(prompt, max_tokens=150, stop=["<|end|>"])
                        answer = output['choices'][0]['text'].strip()

                        st.subheader("Answer")
                        st.success(answer)

                        with st.expander("See the context provided to the AI"):
                            st.markdown(context)
                else:
                    st.warning("Please enter a question.")
