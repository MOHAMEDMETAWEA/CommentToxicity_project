import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import gradio as gr

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_FEATURES = 200000  # Must match training
MAX_SEQUENCE_LENGTH = 1800
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv')
MODEL_PATH_H5 = os.path.join(BASE_DIR, 'toxicity.h5')
MODEL_PATH_KERAS = os.path.join(BASE_DIR, 'toxicity.keras')

def load_and_prepare_resources():
    """Load model and prepare vectorizer"""
    print("🔄 Loading resources...")

    # 1. Setup Vectorizer
    # We need to adapt the vectorizer to the training data to get the same vocabulary
    print(f"📖 Loading dataset from {DATA_PATH} to adapt vectorizer...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Needed for vectorizer.")
    
    df = pd.read_csv(DATA_PATH)
    X = df['comment_text']
    
    print("⚙️ Adapting vectorizer (this may take a moment)...")
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        output_mode='int'
    )
    vectorizer.adapt(X.values)
    print("✅ Vectorizer adapted.")

    # 2. Load Model
    # Check which model file exists
    if os.path.exists(MODEL_PATH_H5):
        load_path = MODEL_PATH_H5
    elif os.path.exists(MODEL_PATH_KERAS):
        load_path = MODEL_PATH_KERAS
    else:
        raise FileNotFoundError("❌ No trained model file found (looked for toxicity.h5 or toxicity.keras)")

    print(f"🧠 Loading model from {load_path}...")
    model = tf.keras.models.load_model(load_path)
    print("✅ Model loaded.")

    return model, vectorizer

# Load resources once at startup
try:
    model, vectorizer = load_and_prepare_resources()
except Exception as e:
    print(f"❌ Error initializing app: {e}")
    exit(1)

def score_comment(comment):
    """Predict toxicity scores for a comment"""
    if not comment:
        return {label: 0.0 for label in LABEL_COLS}
        
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment, verbose=0)
    
    return {label: float(score) for label, score in zip(LABEL_COLS, results[0])}

# =============================================================================
# GRADIO INTERFACE
# =============================================================================
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(
        label="Enter Comment", 
        placeholder="Type a comment here...", 
        lines=3
    ),
    outputs=gr.Label(
        label="Toxicity Analysis", 
        num_top_classes=6
    ),
    title="🛡️ Comment Toxicity Analyzer",
    description="Real-time multi-label classification for toxic comment detection.",
    examples=[
        ["You differ from my opinion, but I respect that."],
        ["You are incredibly annoying and I hope you fail."],
        ["This is a neutral sentence."]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("🚀 Launching Gradio interface...")
    interface.launch(share=True)
