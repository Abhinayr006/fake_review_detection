import pandas as pd
import json
import numpy as np

# Paths
DATA_PATH_YELP_REVIEW = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
DATA_PATH_YELP_USER = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

def count_labels_before_balancing(review_path, user_path, num_reviews=500000):
    """
    Count deceptive and genuine labels before balancing.
    """
    print("Loading user data...")
    user_data = {}
    with open(user_path, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            is_elite_status = 1 if len(user.get('elite', '')) > 0 else 0
            user_data[user['user_id']] = {
                'user_review_count': user.get('review_count', 0),
                'user_average_stars': user.get('average_stars', 0.0),
                'is_elite': is_elite_status,
                'user_friends': len(user.get('friends', [])),
                'user_fans': user.get('fans', 0),
                'user_compliment_count': user.get('compliment_hot', 0) + user.get('compliment_more', 0) + user.get('compliment_profile', 0) + user.get('compliment_cute', 0) + user.get('compliment_list', 0) + user.get('compliment_note', 0) + user.get('compliment_plain', 0) + user.get('compliment_cool', 0) + user.get('compliment_funny', 0) + user.get('compliment_writer', 0) + user.get('compliment_photos', 0)
            }

    print("Loading and sampling review data...")
    reviews = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_reviews:
                break
            try:
                review = json.loads(line)
                reviews.append(review)
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(reviews)
    
    print("Merging with user data and extracting features...")
    df['user_features'] = df['user_id'].map(user_data)
    df.dropna(subset=['user_features'], inplace=True)
    
    df['user_review_count'] = df['user_features'].apply(lambda x: x['user_review_count'])
    df['user_average_stars'] = df['user_features'].apply(lambda x: x['user_average_stars'])
    df['is_elite'] = df['user_features'].apply(lambda x: x['is_elite'])
    df['review_length'] = df['text'].apply(lambda x: len(x.split()))
    df['user_friends'] = df['user_features'].apply(lambda x: x['user_friends'])
    df['user_fans'] = df['user_features'].apply(lambda x: x['user_fans'])
    df['user_compliment_count'] = df['user_features'].apply(lambda x: x['user_compliment_count'])
    
    df.drop(columns=['user_features', 'user_id', 'business_id', 'date', 'review_id'], inplace=True)
    
    # --- New Heuristic for Labeling (does not use training features) ---
    df['deceptive'] = 0
    df.loc[(df['stars'].isin([1, 5])) & (df['review_length'] < 10), 'deceptive'] = 1
    df.loc[df['text'].str.contains('best ever', case=False) & (df['stars'] == 5), 'deceptive'] = 1
    
    deceptive_count = df['deceptive'].sum()
    genuine_count = len(df) - deceptive_count
    
    return deceptive_count, genuine_count, len(df)

if __name__ == "__main__":
    print("Counting deceptive labels before balancing...")
    deceptive, genuine, total = count_labels_before_balancing(DATA_PATH_YELP_REVIEW, DATA_PATH_YELP_USER, num_reviews=500000)
    print(f"Total reviews processed: {total}")
    print(f"Deceptive reviews: {deceptive}")
    print(f"Genuine reviews: {genuine}")
    print(f"Deceptive percentage: {deceptive / total * 100:.2f}%")
