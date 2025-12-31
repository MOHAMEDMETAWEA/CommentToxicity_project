# 🛡️ Comment Toxicity Classification

A deep learning-based **multi-label text classification** system that detects and classifies toxic comments into six different toxicity categories using a Bidirectional LSTM neural network.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Toxicity Categories](#toxicity-categories)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Interactive Demo](#interactive-demo)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated toxic comment classification system that can analyze text and identify multiple types of toxicity simultaneously. Unlike binary classification, this model performs **multi-label classification**, meaning a single comment can be flagged for multiple toxicity types at once.

The model is trained on the **Jigsaw Toxic Comment Classification Challenge** dataset, containing approximately 160,000 human-labeled comments from Wikipedia talk pages.

## ✨ Features

- 🧠 **Bidirectional LSTM Architecture** - Captures context from both past and future words
- 🏷️ **Multi-Label Classification** - Detects multiple toxicity types simultaneously
- 📊 **Comprehensive Evaluation** - Includes AUC-ROC, F1-score, precision, recall, and confusion matrices
- 🌐 **Interactive Web Demo** - Gradio-powered interface for real-time toxicity analysis
- 📈 **Training Visualization** - Detailed plots of training progress and metrics
- 💾 **Model Persistence** - Save and load trained models for deployment

## ⚠️ Toxicity Categories

The model identifies **6 types of toxicity**:

| Label | Description |
| :--- | :--- |
| **toxic** | Generally rude, disrespectful, or unreasonable language |
| **severe_toxic** | Very hateful, aggressive, or extremely offensive content |
| **obscene** | Contains explicit, vulgar, or profane language |
| **threat** | Contains threatening language or intimidation |
| **insult** | Insulting, demeaning, or condescending language |
| **identity_hate** | Targets race, religion, gender, nationality, or other identity |

## 📁 Project Structure

```text
CommentToxicity_project/
├── Toxicity.ipynb                              # Main Jupyter notebook with full pipeline
├── app.py                                      # Standalone Gradio application script
├── toxicity.h5                                 # Trained model weights
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
└── jigsaw-toxic-comment-classification-challenge/
    ├── train.csv/                              # Training data (~160k comments)
    ├── test.csv/                               # Test data
    ├── test_labels.csv/                        # Test labels
    └── sample_submission.csv/                  # Submission format
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU support (recommended for training)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/CommentToxicity_project.git
   cd CommentToxicity_project
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| TensorFlow | 2.x | Deep learning framework |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical operations |
| Matplotlib | Latest | Basic visualizations |
| Seaborn | Latest | Statistical visualizations |
| Scikit-learn | Latest | Metrics and evaluation |
| Gradio | Latest | Interactive web interface |

## 📊 Dataset

This project uses the **[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** dataset from Kaggle.

### Dataset Statistics

- **Training samples**: ~160,000 comments
- **Features**: Raw text comments
- **Labels**: 6 binary toxicity indicators

### Class Distribution

The dataset is **highly imbalanced**, with toxic comments being a minority:

| Category | Count | Percentage |
| :--- | :--- | :--- |
| toxic | ~15,000 | ~9.5% |
| severe_toxic | ~1,600 | ~1.0% |
| obscene | ~8,500 | ~5.3% |
| threat | ~500 | ~0.3% |
| insult | ~7,800 | ~4.9% |
| identity_hate | ~1,400 | ~0.9% |

## 🧠 Model Architecture

The model uses a **Sequential architecture** with the following layers:

```text
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│                 (Text sequences, length=1800)               │
├─────────────────────────────────────────────────────────────┤
│                    EMBEDDING LAYER                          │
│        (vocab_size=200001, embedding_dim=32)                │
├─────────────────────────────────────────────────────────────┤
│                BIDIRECTIONAL LSTM                           │
│              (64 units total, tanh activation)              │
├─────────────────────────────────────────────────────────────┤
│                 BATCH NORMALIZATION                         │
├─────────────────────────────────────────────────────────────┤
│                    DENSE (128 units)                        │
│                 + Dropout (30%)                             │
├─────────────────────────────────────────────────────────────┤
│                    DENSE (256 units)                        │
│                 + Dropout (30%)                             │
├─────────────────────────────────────────────────────────────┤
│                    DENSE (128 units)                        │
│                 + Dropout (20%)                             │
├─────────────────────────────────────────────────────────────┤
│                    OUTPUT LAYER                             │
│           (6 units, sigmoid activation)                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Embedding Layer**: Converts word indices to dense 32-dimensional vectors
- **Bidirectional LSTM**: Processes sequences in both directions for better context understanding
- **BatchNormalization**: Stabilizes and accelerates training
- **Dense + Dropout**: Feature extraction with regularization to prevent overfitting
- **Sigmoid Output**: Enables independent probability for each toxicity class

## 🏋️ Training

### Configuration

| Parameter | Value |
| :--- | :--- |
| Batch Size | 16 |
| Max Vocabulary | 200,000 |
| Sequence Length | 1,800 |
| Epochs | 5 (with early stopping) |
| Learning Rate | 0.001 (Adam) |
| Train/Val/Test Split | 70% / 20% / 10% |

### Callbacks

- **EarlyStopping**: Monitors `val_auc`, patience=2, restores best weights
- **ReduceLROnPlateau**: Reduces learning rate by 50% when validation loss plateaus

### Training Pipeline

1. **Text Vectorization**: Convert raw text to integer sequences
2. **Dataset Creation**: Use `tf.data.Dataset` with caching, shuffling, batching, and prefetching
3. **Model Training**: Fit with validation monitoring
4. **Model Saving**: Export to `.keras` format

## 📈 Evaluation

The model is evaluated using multiple metrics optimized for imbalanced multi-label classification:

### Metrics

- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve (per-class and macro-averaged)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Classification Report**: Detailed per-class metrics
- **Confusion Matrices**: Visual representation of predictions vs. ground truth

## 💻 Usage

### Standalone App

Run the interactive Gradio interface directly without opening a notebook:

```bash
python app.py
```

### Jupyter Notebook

Open `Toxicity.ipynb` in Jupyter Notebook or JupyterLab to run the complete pipeline:

```bash
jupyter notebook Toxicity.ipynb
```

### Basic Usage Support

If implementing custom prediction logic:

```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Load the trained model
model = tf.keras.models.load_model('toxicity.h5')

# Setup vectorizer (must be adapted on same data as training)
vectorizer = TextVectorization(max_tokens=200000, output_sequence_length=1800)
# vectorizer.adapt(training_texts)  # Adapt on training data

# Make predictions
text = "Your comment text here"
vectorized_text = vectorizer([text])
predictions = model.predict(vectorized_text)
```

## 🌐 Interactive Demo

The project includes a **Gradio-powered web interface** for real-time toxicity analysis.

### Launch the Demo

```python
import gradio as gr

# Define scoring function
def score_comment(comment):
    vectorized = vectorizer([comment])
    results = model.predict(vectorized, verbose=0)
    return {label: float(score) for label, score in zip(labels, results[0])}

# Create and launch interface
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(label="Enter Comment", lines=3),
    outputs=gr.Label(label="Toxicity Scores", num_top_classes=6),
    title="🛡️ Comment Toxicity Analyzer",
    theme=gr.themes.Soft()
)
interface.launch(share=True)
```

### Demo Features

- 📝 Text input for comments
- 📊 Real-time toxicity scoring
- 🎨 Clean, modern interface
- 🔗 Sharable public URL option

## 📊 Results

The model achieves strong performance on the test set, particularly for the more common toxicity categories. Performance metrics are visualized through:

- **Training History Plots**: Loss, accuracy, AUC, precision/recall over epochs
- **Confusion Matrices**: Per-class visualization of true vs. predicted labels
- **Classification Report**: Detailed precision, recall, and F1-score breakdown

### Expected Performance

| Metric | Score |
| :--- | :--- |
| Overall Macro AUC-ROC | ~0.95+ |
| Overall Macro F1-Score | Varies by class imbalance |
| Binary Accuracy | ~0.98+ |

> **Note**: Actual performance depends on training conditions and random initialization.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Jigsaw/Google** for providing the Toxic Comment Classification dataset
- **TensorFlow/Keras** team for the deep learning framework
- **Gradio** for the simple yet powerful UI library
- **Kaggle** community for inspiration and insights

---

<p align="center">
  Made with ❤️ for a safer online community
</p>
