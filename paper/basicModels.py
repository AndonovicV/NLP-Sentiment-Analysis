# Basic model to show how to read data using nltk. 
# Show the first 5 rows. 
# Show the number of reviews for each score.
# Show a review example.
# Tokenize the example.
# Tag the tokens.
# Chunk the tagged tokens.
##############################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk

# Read in data
df = pd.read_csv('input/amazon-fine-food-reviews/downsizedReviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)
df.head() # Show the first 5 rows

# Shows the number of reviews for each score
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# Shows a review example. In this case, the review says: "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go."
example = df['Text'][50]
print(example)

# Tokenize the example
tokens = nltk.word_tokenize(example)
print (tokens[:10])

# Tag the tokens
tagged = nltk.pos_tag(tokens)
print (tagged[:10])

# Chunk the tagged tokens
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# Vader Sentiment Analysis
# VADER is a tool for sentiment analysis that looks at the words in a sentence, checks if they’re positive, negative, or neutral, and adds up the scores to figure out the overall sentiment. 
# It’s super quick and simple, but it doesn’t really get the context or relationships between words, which is something humans naturally pick up on. For example this Vador model would be very bad with sarcasm.
# Stop words are also removed (and, the, is, etc.) because they don’t really add any sentiment to the sentence. 
##############################################################################################################
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm # Progress bar tracker

sia = SentimentIntensityAnalyzer() # sia is the SentimentIntensityAnalyzer object

# precentage of positive, negative, neutral, and compound sentiment
print(sia.polarity_scores('I am so happy!')) # {'neg': 0.0, 'neu': 0.318, 'pos': 0.682, 'compound': 0.6468}
print(sia.polarity_scores('This is the worst thing ever.')) # {'neg': 0.451, 'neu': 0.549, 'pos': 0.0, 'compound': -0.6249}

# Sia example of the example review
print(sia.polarity_scores(example)) # {'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}



# Run the polarity score on the entire dataset
# A polarity score in NLP measures the overall sentiment of text, indicating how positive, negative, or neutral it is.
res = {} # result dictionary
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Using pandas dataframe to store the results (easier to work with)
vaders = pd.DataFrame(res).T # .T flips everything horizontally
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

# Now we have sentiment score and metadata
print(vaders.head())

# Plot the compound score by Amazon Star Review
# Checking to see how accurate the sentiment analysis is by comparing the compound score to the Amazon star review (Score).
# This shows for example that 1 star review has lower compound score, and 5 star review has higher. Which is exactly as expected.
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

# Plot the Vader Negative, Neutral and Positive score by Amazon Star Review
# Checking to see how accurate the sentiment analysis is by comparing the neg, neu, pos score to the Amazon star review (Score).
# Aditionaly shows that positivity is higher as score is higher, the neutral is flat and negiative goes down (it becomes less negative of a comment as the star becomes higher).
# This confirms that vador is valuable in having a connection between the sentiment score and the actual review from Amazon.
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# ROBERTA Pretained Model
# Since Vader is not good with sarcasm for example, we can use a transformer based deep learning model to get a more accurate sentiment analysis.
# This is a deep learning model that was pre-trained on a lot of data from twitter comments that were labelled, and is provided by a hugging face.
# Essencialy this means that we don't have to train the model, we can just use this trained weights and apply it to our database.
##############################################################################################################
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Load the tokenizer and model

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" # if run first time, it will download the model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# VADER results on example (just to compare)
print(example)
print(sia.polarity_scores(example)) # {'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict) # {'roberta_neg': 0.97635514, 'roberta_neu': 0.020687465, 'roberta_pos': 0.0029573678}
# Here we can see that the Roberta model is much more accurate than the Vader model. 
# The Vader model said that the review was more neutral (77%), but the Roberta model says that the review is very negative (97%).

# Run the Roberta model on the entire dataset
# Create a function to make it easier to run



def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError: # This error occurs when the review is too long for the model to handle. So we just skip it and mark down the id of the skipped review (In case we really need to know which review was skipped).
        print(f'Broke for id {myid}')

# Using pandas dataframe to store and compare the results
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
print(results_df.head())

# Plot Vador and Roberta poutput score by Amazon Star Review
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()
# Here we can see that even though there is small corelation between the vador and roberta model. 
# But one thing that becomes very clear is that the Vador model is a little bit less confident in all of it's predictions compared the Roberta model, which really separates the negative, neutral and positive scores.
# And comparing Robrta to the actual amazon Scores, we can see that the Roberta model is much more accurate than the Vader model. Especially in the 1 star and 5 star reviews.



# REVIEW EXAMPLES
# Now that we have the sentiment scores and the 5 star amazon ranking of the review. We can look and see where the model does the opposite of what we think it should.
# To do that we can query where there is a 1 star review, but the model thinks it's a positive review.

# Find the highest positive score where the ranking is 1 star'
print(results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]) # So a text that is said to be positive by the model, but is actually a 1 star review.

print(results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0])

# Now we can do it vice versa, where the model thinks it's a negative review, but the ranking is 5 stars.
# Interesting enoguh, this time the output is the same for both models.

print(results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0])

print(results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0])


# Best part: We can just use the model using the hugging face transformers pipelines in literaly two lines of code
from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis") # this will automatically dowmload their default model for sentiment analysis

print(sent_pipeline("This label is not horrible, but it's not too good either.")) # [{'label': 'POSITIVE', 'score': 0.9998733997344971}]
print(sent_pipeline("This is the worst thing ever.")) # [{'label': 'NEGATIVE', 'score': 0.999747097492218}]

print(example) # This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.
print(sent_pipeline(example)) # [{'label': 'NEGATIVE', 'score': 0.9994776844978333}]