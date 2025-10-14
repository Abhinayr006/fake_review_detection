# data_preprocessing_yelp.py

import pandas as pd
import json
import numpy as np
import re

def load_yelp_data(review_path, user_path, num_reviews=1000000):
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
            # Extract user behavioral features from yelp_academic_dataset_user.json....................
            user_data[user['user_id']] = {
                'user_review_count': user.get('review_count', 0),  # Total number of reviews written by this user
                'user_average_stars': user.get('average_stars', 0.0),  # User's average star rating across all their reviews
                'is_elite': is_elite_status,  # 1 if user has elite status in current or past years, 0 otherwise
                'user_friends': len(user.get('friends', [])),  # Number of friends this user has on Yelp
                'user_fans': user.get('fans', 0),  # Number of fans this user has
                'user_compliment_count': user.get('compliment_hot', 0) + user.get('compliment_more', 0) + user.get('compliment_profile', 0) + user.get('compliment_cute', 0) + user.get('compliment_list', 0) + user.get('compliment_note', 0) + user.get('compliment_plain', 0) + user.get('compliment_cool', 0) + user.get('compliment_funny', 0) + user.get('compliment_writer', 0) + user.get('compliment_photos', 0)  # Sum of all compliment types received by this user
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
    
    # Extract behavioral features from user data for multi-modal training.......................
    df['user_review_count'] = df['user_features'].apply(lambda x: x['user_review_count'])  # Total reviews by user (from user.json)
    df['user_average_stars'] = df['user_features'].apply(lambda x: x['user_average_stars'])  # User's average rating across all reviews (from user.json)
    df['is_elite'] = df['user_features'].apply(lambda x: x['is_elite'])  # Binary: 1 if user has elite status, 0 otherwise (derived from elite field in user.json)
    df['review_length'] = df['text'].apply(lambda x: len(x.split()))  # Number of words in the review text (calculated from review text)
    df['user_friends'] = df['user_features'].apply(lambda x: x['user_friends'])  # Number of friends the user has (length of friends list from user.json)
    df['user_fans'] = df['user_features'].apply(lambda x: x['user_fans'])  # Number of fans the user has (from user.json)
    df['user_compliment_count'] = df['user_features'].apply(lambda x: x['user_compliment_count'])  # Total compliments received (sum of all compliment types from user.json)
    
    df.drop(columns=['user_features', 'user_id', 'business_id', 'date', 'review_id'], inplace=True)
    
    # --- New Heuristic for Labeling (does not use training features) ---
    df['deceptive'] = 0
    df.loc[(df['stars'].isin([1, 5])) & (df['review_length'] < 10), 'deceptive'] = 1
    df.loc[df['text'].str.contains('best ever', case=False) & (df['stars'] == 5), 'deceptive'] = 1
    
    df_deceptive = df[df['deceptive'] == 1]
    df_genuine = df[df['deceptive'] == 0]
    
    # Explicitly create balanced dataset with equal number of deceptive and genuine reviews
    if not df_deceptive.empty and not df_genuine.empty:
        n_samples = min(len(df_deceptive), len(df_genuine))
        df_deceptive_sampled = df_deceptive.sample(n=n_samples, random_state=42)
        df_genuine_sampled = df_genuine.sample(n=n_samples, random_state=42)
        df = pd.concat([df_deceptive_sampled, df_genuine_sampled]).sample(frac=1, random_state=42)
    else:
        print("Warning: No deceptive or genuine reviews found with the current heuristic.")
        
    return df[['text', 'deceptive', 'user_review_count', 'user_average_stars', 'is_elite', 'review_length', 'user_friends', 'user_fans', 'user_compliment_count']]
