# data_preprocessing_yelp.py

import pandas as pd
import json
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer

def load_yelp_data(review_path, user_path, num_reviews=500000):
    """
    Loads a limited number of reviews and user data from the Yelp dataset
    to create a comprehensive dataframe with both textual and behavioral features.
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

    # Advanced feature engineering
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['num_sentences'] = df['text'].apply(lambda x: len(re.split(r'[.!?]+', x)))
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)

    df.drop(columns=['user_features', 'user_id', 'business_id', 'date', 'review_id'], inplace=True)
    
    # --- Improved Heuristic for Labeling (does not use training features) ---
    df['deceptive'] = 0

    # Original heuristics
    df.loc[(df['stars'].isin([1, 5])) & (df['review_length'] < 10), 'deceptive'] = 1
    df.loc[df['text'].str.contains('best ever', case=False) & (df['stars'] == 5), 'deceptive'] = 1

    # Additional heuristics for better detection
    # Excessive punctuation
    df.loc[df['text'].str.count('!') > 5, 'deceptive'] = 1
    df.loc[df['text'].str.count('\\?') > 3, 'deceptive'] = 1

    # All caps words (more than 30% of words are all caps)
    df['caps_ratio'] = df['text'].apply(lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1) / len(x.split()) if len(x.split()) > 0 else 0)
    df.loc[df['caps_ratio'] > 0.3, 'deceptive'] = 1

    # Suspicious phrases with high stars and short length
    suspicious_phrases = ['amazing', 'incredible', 'perfect', 'fantastic', 'awesome', 'love it', 'best place ever']
    for phrase in suspicious_phrases:
        df.loc[df['text'].str.contains(phrase, case=False) & (df['stars'] == 5) & (df['review_length'] < 15), 'deceptive'] = 1

    # Low user activity but high stars
    df.loc[(df['user_review_count'] < 5) & (df['stars'] == 5) & (df['review_length'] < 20), 'deceptive'] = 1

    # Very short reviews with extreme ratings
    df.loc[(df['stars'].isin([1, 5])) & (df['review_length'] < 5), 'deceptive'] = 1

    # Drop the temporary caps_ratio column
    df.drop(columns=['caps_ratio'], inplace=True)
    
    df_deceptive = df[df['deceptive'] == 1]
    df_genuine = df[df['deceptive'] == 0]
    
    if not df_deceptive.empty and len(df_deceptive) < len(df_genuine):
        df_genuine_sampled = df_genuine.sample(n=len(df_deceptive), random_state=42)
        df = pd.concat([df_deceptive, df_genuine_sampled]).sample(frac=1, random_state=42)
    elif not df_deceptive.empty:
        df = pd.concat([df_deceptive, df_genuine]).sample(frac=1, random_state=42)
    else:
        print("Warning: No deceptive reviews found with the current heuristic.")
        
    return df