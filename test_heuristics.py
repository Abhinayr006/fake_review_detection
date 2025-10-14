import pandas as pd
from data_preprocessing_yelp import load_yelp_data

# Test the updated heuristics with a small sample
review_path = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
user_path = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

# Load a small sample to test
df = load_yelp_data(review_path, user_path, num_reviews=10000)

# Count deceptive vs genuine
deceptive_count = df['deceptive'].sum()
genuine_count = len(df) - deceptive_count

print(f"Total reviews processed: {len(df)}")
print(f"Deceptive reviews: {deceptive_count}")
print(f"Genuine reviews: {genuine_count}")
print(".1f")

# Show some examples
print("\nSample deceptive reviews:")
deceptive_samples = df[df['deceptive'] == 1].head(3)
for idx, row in deceptive_samples.iterrows():
    print(f"Text: {row['text'][:100]}...")
    print(f"Length: {row['review_length']}")
    print("---")
