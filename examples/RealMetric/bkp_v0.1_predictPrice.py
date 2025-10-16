# ==============================================================================
# 1. PREREQUISITE INSTALLATION
# ==============================================================================
import subprocess
import sys
import pkg_resources

# List of required packages
required_packages = {
    'streamlit',
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'llama-cpp-python',
    'huggingface_hub'
}

# Check for missing packages
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = required_packages - installed_packages

# Install missing packages
if missing_packages:
    print(f"Missing packages: {', '.join(missing_packages)}. Installing...")
    try:
        # Using subprocess to ensure pip is run correctly
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
        print("All required packages are now installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        print("Please install the missing packages manually using 'pip install'.")
        sys.exit(1)
    finally:
        # This 'finally' block will execute regardless of whether the installation succeeded or failed.
        print("Prerequisite check complete.")

# Now, import the libraries
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import os
except ImportError as e:
    st.error(f"Failed to import a necessary library: {e}. Please ensure all packages are installed correctly.")
    sys.exit(1)

# ==============================================================================
# 2. SETUP & CONFIGURATION
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Real Estate AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- File Paths ---
DATA_FOLDER = "Data"
DATA_FILE = "listings_sample.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

# --- Model Configuration ---
# Using a small, capable model that runs on CPU
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = None # Will be set after download

# ==============================================================================
# 3. CACHED HELPER FUNCTIONS (for performance)
# ==============================================================================

@st.cache_data
def load_and_clean_data(file_path):
    """Loads data, cleans it, and engineers features."""
    if not os.path.exists(file_path):
        st.error(f"Data file not found at '{file_path}'. Please make sure it exists.")
        return None

    try:
        df = pd.read_csv(file_path)
        
        # Basic cleaning
        df.dropna(subset=['price', 'sqft', 'beds', 'baths', 'remarks'], inplace=True)
        
        # Convert to numeric, coercing errors
        for col in ['price', 'sqft', 'beds', 'baths']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where conversion failed
        df.dropna(subset=['price', 'sqft', 'beds', 'baths'], inplace=True)
        
        # Feature Engineering: Price per SqFt
        # Avoid division by zero
        df = df[df['sqft'] > 0].copy()
        df['price_per_sqft'] = df['price'] / df['sqft']
        
        # Ensure remarks are strings
        df['remarks'] = df['remarks'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"An error occurred while loading or cleaning the data: {e}")
        return None

@st.cache_resource
def get_llm_model():
    """Downloads and loads the GGUF model."""
    with st.spinner(f"Downloading LLM model: {MODEL_FILE}... (this may take a while the first time)"):
        try:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        except Exception as e:
            st.error(f"Failed to download model. Check your internet connection. Error: {e}")
            return None
    
    with st.spinner("Loading model into memory..."):
        try:
            # IMPORTANT: Set embedding=True for RAG task
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,       # Context window
                n_gpu_layers=0,  # Set to 0 for CPU-only
                verbose=False,
                embedding=True   # Crucial for creating embeddings
            )
            return llm
        except Exception as e:
            st.error(f"Failed to load the LLM model. Error: {e}")
            return None

@st.cache_resource
def train_prediction_model(_df):
    """Trains a RandomForestRegressor to predict price."""
    features = ['beds', 'baths', 'sqft', 'city']
    target = 'price'
    
    X = _df[features]
    y = _df[target]
    
    # Preprocessing: One-hot encode the 'city' column
    categorical_features = ['city']
    numeric_features = ['beds', 'baths', 'sqft']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the pipeline with the preprocessor and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split data and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model_pipeline, mae, r2

@st.cache_data
def create_embeddings(_llm, remarks_list):
    """Creates embeddings for a list of text documents using the LLM."""
    with st.spinner("Creating text embeddings for RAG..."):
        try:
            embeddings = np.array([_llm.create_embedding(text)['data'][0]['embedding'] for text in remarks_list])
            return embeddings
        except Exception as e:
            st.error(f"Failed to create embeddings. Error: {e}")
            return None

# ==============================================================================
# 4. UI SECTIONS
# ==============================================================================

def display_data_exploration(df):
    st.header("1. Data Exploration & Cleaning")
    st.write("The raw data is loaded, cleaned (missing values removed, data types corrected), and a `price_per_sqft` column is added.")
    
    st.subheader("Cleaned Dataset")
    st.dataframe(df.head())
    
    st.subheader("Dataset Summary")
    st.write(df.describe())
    
    st.subheader("Visual Comparisons")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Average Price by City")
        avg_price_city = df.groupby('city')['price'].mean().sort_values()
        fig1, ax1 = plt.subplots()
        avg_price_city.plot(kind='barh', ax=ax1, color=sns.color_palette("viridis", len(avg_price_city)))
        ax1.set_xlabel("Average Price ($)")
        ax1.set_title("Average Listing Price per City")
        st.pyplot(fig1)
        
    with col2:
        st.write("#### Price Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax2, color='skyblue')
        ax2.set_xlabel("Price ($)")
        ax2.set_title("Distribution of Listing Prices")
        st.pyplot(fig2)

def display_price_prediction(df):
    st.header("2. Price Prediction Model")
    st.info("""
    **Model Used:** `RandomForestRegressor` from Scikit-learn.
    
    This model is trained to predict the `price` of a listing based on its features: number of beds, baths, square footage, and city.
    """)
    
    model, mae, r2 = train_prediction_model(df)
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    col2.metric("R-squared (R²)", f"{r2:.2f}")
    st.write("MAE indicates the average error in price prediction. R² represents the proportion of the variance in price that is predictable from the features.")
    
    st.subheader("Try the Predictor")
    with st.form("prediction_form"):
        beds = st.slider("Number of Bedrooms", 1, df['beds'].max(), 3)
        baths = st.slider("Number of Bathrooms", 1.0, float(df['baths'].max()), 2.0, 0.5)
        sqft = st.number_input("Square Footage", 500, df['sqft'].max(), 1500)
        city = st.selectbox("City", df['city'].unique())
        
        submit_button = st.form_submit_button("Predict Price")
        
        if submit_button:
            # Create a DataFrame for the single prediction
            input_data = pd.DataFrame([[beds, baths, sqft, city]], columns=['beds', 'baths', 'sqft', 'city'])
            predicted_price = model.predict(input_data)[0]
            st.success(f"Predicted Price: **${predicted_price:,.2f}**")

def display_llm_tools(df, llm):
    st.header("3. LLM Integration")
    st.info(f"""
    **Model Used:** `{MODEL_REPO}` (a TinyLlama GGUF model).
    
    This is a small, open-source Large Language Model running locally on your machine's CPU. It's used for both generating summaries and answering questions about the listings.
    """)
    
    tool_choice = st.radio("Choose an LLM Tool:", ("RAG Q&A over Remarks", "Auto-Summary Generator"), horizontal=True)
    
    if tool_choice == "RAG Q&A over Remarks":
        display_rag_qa(df, llm)
    else:
        display_summarizer(df, llm)

def display_rag_qa(df, llm):
    st.subheader("RAG Mini-Q&A over Listing Remarks")
    st.write("Ask a question about the property descriptions (e.g., 'Which homes have a large backyard?' or 'Are there any pet-friendly condos?'). The system will find the most relevant listings and use them to answer your question.")
    
    # 1. Create embeddings
    remarks_list = df['remarks'].tolist()
    embeddings = create_embeddings(llm, remarks_list)
    
    if embeddings is not None:
        # 2. Get user query and embed it
        user_query = st.text_input("Your Question:", "Which properties have a garage?")
        
        if st.button("Get Answer"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching and generating answer..."):
                    query_embedding = np.array(llm.create_embedding(user_query)['data'][0]['embedding']).reshape(1, -1)
                    
                    # 3. Find top-k similar remarks (Alternative to FAISS)
                    similarities = cosine_similarity(query_embedding, embeddings)[0]
                    top_k_indices = np.argsort(similarities)[-3:][::-1] # Top 3
                    
                    # 4. Build context and prompt
                    context = ""
                    st.write("#### Most Relevant Listings Found:")
                    for i, idx in enumerate(top_k_indices):
                        context += f"Listing {i+1} Remarks: {df.iloc[idx]['remarks']}\n\n"
                        with st.expander(f"Listing {i+1} (Similarity: {similarities[idx]:.2f})"):
                            st.write(df.iloc[idx])
                    
                    prompt_template = f"""
                    <|system|>
                    You are a helpful real estate assistant. Answer the user's question based *only* on the context provided from the listings. Do not use any outside knowledge. If the answer is not in the context, say so.</s>
                    <|user|>
                    CONTEXT:
                    ---
                    {context}
                    ---
                    QUESTION: {user_query}</s>
                    <|assistant|>
                    """
                    
                    # 5. Generate response
                    response = llm(prompt=prompt_template, max_tokens=256, stop=["<|user|>"])
                    answer = response['choices'][0]['text']
                    
                    st.success("#### Answer:")
                    st.write(answer)

def display_summarizer(df, llm):
    st.subheader("Auto-Summary Generator")
    st.write("Select a listing to automatically generate a 2-3 sentence summary using its key details.")
    
    listing_index = st.slider("Select a Listing to Summarize:", 0, len(df)-1, 0)
    
    selected_listing = df.iloc[listing_index]
    
    st.write("#### Selected Listing Details:")
    st.write(selected_listing)
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            # Create a detailed prompt with structured data
            prompt_template = f"""
            <|system|>
            You are a real estate copywriter. Generate a concise, appealing 2-3 sentence summary for the following property.</s>
            <|user|>
            Generate a summary based on these facts:
            - City: {selected_listing['city']}
            - Price: ${selected_listing['price']:,.0f}
            - Beds: {selected_listing['beds']}
            - Baths: {selected_listing['baths']}
            - SqFt: {selected_listing['sqft']}
            - Remarks: "{selected_listing['remarks']}"</s>
            <|assistant|>
            """
            
            response = llm(prompt=prompt_template, max_tokens=150, stop=["<|user|>"])
            summary = response['choices'][0]['text']
            
            st.success("#### Generated Summary:")
            st.write(summary)

# ==============================================================================
# 5. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    st.title("🏡 Real Estate AI Assistant")
    
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a section:",
        ["Data Exploration", "Price Prediction", "LLM Tools"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This app demonstrates data analysis, predictive modeling, and LLM integration on a real estate dataset.")

    # --- Load Data ---
    df = load_and_clean_data(DATA_PATH)
    
    if df is None:
        st.stop() # Stop execution if data loading failed

    # --- Page Routing ---
    if app_mode == "Data Exploration":
        display_data_exploration(df)
    elif app_mode == "Price Prediction":
        display_price_prediction(df)
    elif app_mode == "LLM Tools":
        llm = get_llm_model()
        if llm is not None:
            display_llm_tools(df, llm)
        else:
            st.error("LLM could not be loaded. The LLM Tools section is unavailable.")

if __name__ == "__main__":
    main()
