import pandas as pd
from datasets import load_dataset
import os

def prepare_data():
    print("Fetching data from Hugging Face...")
    # Pivoting to a verified, stable dataset
    dataset = load_dataset("gretelai/symptom_to_diagnosis")
    
    # Hugging Face datasets usually come in splits. We will grab the 'train' split.
    df = pd.DataFrame(dataset['train'])
    
    print("\n--- Data Overview ---")
    # Display the first 5 rows to see what the raw data looks like
    print(df.head())
    
    print("\n--- Checking for Class Imbalance ---")
    # In this new dataset, the target diagnosis is stored in the 'output_text' column
    class_counts = df['output_text'].value_counts()
    print(class_counts)
    
    # Save the processed data locally
    os.makedirs("data", exist_ok=True)
    save_path = "data/raw_data.csv"
    df.to_csv(save_path, index=False)
    
    print(f"\nStep Complete: Data saved successfully to {save_path}")

if __name__ == "__main__":
    prepare_data()