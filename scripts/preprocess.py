import pandas as pd
import json
import os
from transformers import AutoTokenizer

def preprocess_data():
    print("Loading raw data...")
    df = pd.read_csv("data/raw_data.csv")
    
    # ---------------------------------------------------------
    # STEP 1: Label Encoding (Target Variables)
    # ---------------------------------------------------------
    # Get all unique diagnoses and sort them for consistency
    unique_labels = sorted(df['output_text'].unique())
    
    # Create two dictionaries: Text -> Number, and Number -> Text
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    
    # Apply the mapping to our dataset to create a new column 'labels'
    df['labels'] = df['output_text'].map(label2id)
    
    # Save these dictionaries! We will desperately need id2label during Inference (Day 7)
    with open("data/label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    print(f"Saved Label Dictionary with {len(unique_labels)} unique classes.")

    # ---------------------------------------------------------
    # STEP 2: Tokenization (Input Variables)
    # ---------------------------------------------------------
    # Load the DistilBERT tokenizer. 
    # 'distilbert-base-uncased' means it converts everything to lowercase.
    print("\nLoading DistilBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print("Tokenizing text... (this may take a few seconds)")
    
    # We tokenize the 'input_text' column. 
    # truncation=True: Cuts off text longer than 512 tokens (DistilBERT's limit)
    # padding='max_length': Pads shorter text with 0s so all matrices are the same size
    encoded_inputs = tokenizer(
        df['input_text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128, # 128 is plenty for short symptom descriptions, saves VRAM!
        return_tensors="pt" # Return PyTorch tensors
    )
    
    # We don't save the raw tensors to CSV. Instead, we save a clean dataframe 
    # with the lists of token IDs so we can load it easily into PyTorch later.
    df['input_ids'] = encoded_inputs['input_ids'].tolist()
    df['attention_mask'] = encoded_inputs['attention_mask'].tolist()
    
    # ---------------------------------------------------------
    # STEP 3: Save Processed Data
    # ---------------------------------------------------------
    save_path = "data/processed_data.csv"
    # We only save the columns the model actually needs
    df[['input_ids', 'attention_mask', 'labels']].to_csv(save_path, index=False)
    
    print(f"\nPreprocessing Complete! Saved to {save_path}")
    
    # Let's peek at what the model will actually see:
    print("\n--- What the AI sees (Row 0) ---")
    print("Original Text:", df['input_text'].iloc[0][:50], "...")
    print("Token IDs:", df['input_ids'].iloc[0][:10], "... (padded with 0s)")
    print("Label ID:", df['labels'].iloc[0])

if __name__ == "__main__":
    preprocess_data()