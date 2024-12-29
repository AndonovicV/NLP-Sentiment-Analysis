import pandas as pd

# Load the original CSV file
file_path = 'input/amazon-fine-food-reviews/Reviews.csv' # This was the original file (286 MB)
data = pd.read_csv(file_path)

# Select the first 1000 rows
downsized_data = data.head(1000)

downsized_data.to_csv('downsizedReviews.csv', index=False)

print("The file has been downsized and saved as downsizedReviews.csv.")
