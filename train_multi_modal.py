from data_preprocessing_yelp import load_yelp_data
from multi_modal_model import train_multi_modal_model

# Paths
DATA_PATH_YELP_REVIEW = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
DATA_PATH_YELP_USER = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

if __name__ == "__main__":
    print("Loading Yelp data...")
    df = load_yelp_data(DATA_PATH_YELP_REVIEW, DATA_PATH_YELP_USER, num_reviews=500000)
    print(f"Data loaded: {len(df)} samples")
    train_multi_modal_model(df)
