# data_preprocessing_yelp.py

import pandas as pd
import json
import numpy as np
import re

def load_yelp_data(review_path, user_path, num_reviews=500000): # Change this number
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
                'is_elite': is_elite_status
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
    
    df.drop(columns=['user_features', 'user_id', 'business_id', 'date', 'review_id'], inplace=True)
    
    # --- The New, More Complex Labeling Heuristic ---
    # This makes the classification task more challenging and realistic.
    
    df['suspicious_score'] = 0

    # Factor 1: Extremity of rating from new users
    df.loc[(df['stars'] >= 4) & (df['user_review_count'] <= 1), 'suspicious_score'] += 1
    df.loc[(df['stars'] <= 2) & (df['user_review_count'] <= 1), 'suspicious_score'] += 1

    # Factor 2: Overly enthusiastic or negative language (short reviews)
    df.loc[(df['stars'].isin([1, 5])) & (df['review_length'] < 10), 'suspicious_score'] += 1

    # Create a 'deceptive' label based on a threshold
    df['deceptive'] = (df['suspicious_score'] >= 2).astype(int)

    # Balance the dataset by sampling from the large 'genuine' class
    df_deceptive = df[df['deceptive'] == 1]
    df_genuine = df[df['deceptive'] == 0]
    
    # Sample a number of genuine reviews to match the number of deceptive ones.
    if not df_deceptive.empty and len(df_deceptive) < len(df_genuine):
        df_genuine_sampled = df_genuine.sample(n=len(df_deceptive), random_state=42)
        df = pd.concat([df_deceptive, df_genuine_sampled]).sample(frac=1, random_state=42)
    
    # If the number of deceptive reviews is very low, we might not get enough genuine reviews to sample from.
    # In that case, we can just return the original df.
    elif not df_deceptive.empty:
        df = pd.concat([df_deceptive, df_genuine]).sample(frac=1, random_state=42)
    else:
        print("Warning: No deceptive reviews found with the current heuristic.")
    
    return df