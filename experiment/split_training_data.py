import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def split_training_data(training_file="training.json", test_size=0.2, random_state=42):
    """Split training.json into train and test sets"""
    
    # Load training data
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"Original training data has {len(training_data)} problems")
    
    # Get problem IDs
    problem_ids = list(training_data.keys())
    
    # Split problem IDs
    train_ids, test_ids = train_test_split(
        problem_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train and test splits
    train_data = {pid: training_data[pid] for pid in train_ids}
    test_data = {pid: training_data[pid] for pid in test_ids}
    
    print(f"Train set: {len(train_data)} problems")
    print(f"Test set: {len(test_data)} problems")
    
    # Save splits
    with open('training_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
        
    with open('training_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Also create metadata
    split_info = {
        "total_problems": len(training_data),
        "train_problems": len(train_data), 
        "test_problems": len(test_data),
        "test_size": test_size,
        "random_state": random_state,
        "train_ids": train_ids,
        "test_ids": test_ids
    }
    
    with open('data_split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return train_data, test_data, split_info

def load_extracted_features(features_file="extracted_features.csv"):
    """Load and analyze the extracted features"""
    
    if not Path(features_file).exists():
        print(f"Features file {features_file} not found")
        return None
    
    df = pd.read_csv(features_file)
    print(f"Extracted features shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique problems: {df['prob_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Split the training data
    train_data, test_data, split_info = split_training_data()
    
    # Load and analyze features
    features_df = load_extracted_features()
    
    print("\nData split completed successfully!")
    print(f"Files created: training_train.json, training_test.json, data_split_info.json")