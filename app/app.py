from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

app = Flask(__name__)

# Load model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load dataset
file_path = 'app/dataset/downsizedReviews.csv'
data = pd.read_csv(file_path)
if 'ProductId' not in data.columns or 'Score' not in data.columns:
    raise ValueError("The dataset must contain 'ProductId' and 'Score' columns.")

@app.route('/')
def index():
    product_review_counts = data['ProductId'].value_counts()
    sorted_product_ids = product_review_counts.index.tolist()[:40]
    return render_template('index.html', product_ids=sorted_product_ids)

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.json.get("review", "")
    if not review:
        return jsonify({"error": "Please provide a review"}), 400

    encoded_input = tokenizer(review, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    probabilities = softmax(scores)

    sentiment_scores = {label: float(prob) for label, prob in zip(['Negative', 'Neutral', 'Positive'], probabilities)}
    positivity = sentiment_scores['Positive'] * 100 if sentiment_scores['Positive'] > sentiment_scores['Negative'] else 100 - (sentiment_scores['Negative'] * 100)

    if positivity < 20:
        rating = 1
        text_result = "Very Bad"
    elif 20 <= positivity < 40:
        rating = 2
        text_result = "Bad"
    elif 40 <= positivity < 60:
        rating = 3
        text_result = "Neutral"
    elif 60 <= positivity < 80:
        rating = 4
        text_result = "Good"
    else:
        rating = 5
        text_result = "Very Good"

    return jsonify({"rating": rating, "result": text_result, "raw_output": sentiment_scores})

@app.route('/average_score', methods=['POST'])
def average_score():
    product_id = request.json.get('product_id')
    filtered_data = data[data['ProductId'] == product_id]
    if not filtered_data.empty:
        avg_score = filtered_data['Score'].mean()
        rating = round(avg_score)
        result = f"The actual average score for Product ID {product_id} is {avg_score:.2f}"
    else:
        result = f"No reviews available for Product ID {product_id}."
        rating = 0

    return jsonify({"result": result, "rating": rating})

@app.route('/expected_score', methods=['POST'])
def expected_score():
    product_id = request.json.get('product_id')
    filtered_data = data[data['ProductId'] == product_id]
    if filtered_data.empty:
        return jsonify({"result": f"No reviews available for Product ID {product_id}.", "rating": 0})

    expected_scores = []

    for _, row in filtered_data.iterrows():
        try:
            text = row['Text']
            encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            probabilities = softmax(scores)
            sentiment_scores = {label: float(prob) for label, prob in zip(['Negative', 'Neutral', 'Positive'], probabilities)}

            positivity = sentiment_scores['Positive'] * 100 if sentiment_scores['Positive'] > sentiment_scores['Negative'] else 100 - (sentiment_scores['Negative'] * 100)
            if positivity < 20:
                rating = 1
            elif 20 <= positivity < 40:
                rating = 2
            elif 40 <= positivity < 60:
                rating = 3
            elif 60 <= positivity < 80:
                rating = 4
            else:
                rating = 5

            expected_scores.append(rating)
        except RuntimeError: # This error occurs when the review is too long for the model to handle. 
                             # You can mark down the id of the skipped review (In case we really need to know which review was skipped) by using: print(f'Broke for id {myid}')
            continue

    if expected_scores:
        avg_expected_score = sum(expected_scores) / len(expected_scores)
        result = f"The expected average score for Product ID {product_id} is {avg_expected_score:.2f}"
        rating = round(avg_expected_score)
    else:
        result = f"No valid reviews available for Product ID {product_id}."
        rating = 0

    return jsonify({"result": result, "rating": rating})

if __name__ == '__main__':
    app.run(debug=True)
