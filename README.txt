# Sentiment Analysis Project

This project showcases two key components of sentiment analysis using NLP: a theoretical analysis file (`basicModel.py`) and a practical interactive web application (`app.py`). The project explores sentiment analysis methods using various models and provides a functional interface for real-world business insights.

---

## Components

### 1. **Theoretical Analysis (`basicModel.py`)**
This file focuses on the theoretical and computational side of sentiment analysis. It includes:
- Implementation and comparison of sentiment analysis approaches:
  - **NLTK's Vader**: Suitable for simple polarity-based sentiment scoring.
  - **Roberta (via Hugging Face Transformers)**: A pre-trained transformer model for complex sentiment analysis, including handling sarcasm and nuanced language.
- Tokenization, sentiment scoring, and model evaluation techniques.
- Visualization of model performance through plots (e.g., accuracy, precision, etc.).

Key Deliverable: This forms the basis of the paper, explaining and comparing models with detailed performance metrics.

### 2. **Interactive Web Application (`app.py`)**
This Flask-based web app provides a user-friendly interface to interact with sentiment analysis models. Key features include:
1. **Review Sentiment Analysis**:
   - Users can enter their own review or use example reviews.
   - The model predicts the sentiment and displays the results as star ratings.
2. **Product Review Analysis**:
   - Select a product to compute:
     - **Average Review Score**: The mathematical mean of actual user reviews.
     - **Predicted Model Score**: What the model predicts as the sentiment score for all reviews of the product.
   - Compare actual user ratings with model-predicted ratings.

Practical Insight: The web app demonstrates how businesses can leverage sentiment analysis for understanding product sentiment and aligning model predictions with real-world reviews.

---

## Installation

### Prerequisites
- Python 3.8+

### Steps
Theoretical Analysis: run basicModel.py

Interactive Web App: run python app.py
For the web app, open your browser and navigate to: http://127.0.0.1:5000 (or copy the path from the terminal after running app.py)
