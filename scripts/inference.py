import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MedicalAssistant:
    def __init__(self, model_path="models/medical_model_final"):
        print("Loading Medical AI Model...")
        # 1. Load the Tokenizer (Must be the exact same one used in preprocessing!)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # 2. Load your custom trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move model to GPU if available (your RTX 3050 Ti will handle this easily)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode (turns off dropout layers)
        print(f"Model loaded successfully on {self.device}!")

    def diagnose(self, symptoms: str):
        # Step A: Tokenize the user's input
        inputs = self.tokenizer(
            symptoms, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)

        # Step B: Pass it through the model (No gradients needed for inference = saves memory)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Step C: Get the prediction
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # Step D: Convert the ID back to the English label using the model's config
        predicted_label = self.model.config.id2label[predicted_class_id]
        
        return predicted_label

if __name__ == "__main__":
    # Let's test it!
    assistant = MedicalAssistant()
    
    print("\n--- Testing the Medical AI ---")
    test_symptoms = [
        "I have a severe headache, my neck feels stiff, and light hurts my eyes.",
        "I've been sneezing non-stop, my throat is scratchy, and I have a runny nose.",
        "There is a red, itchy rash on my arm that keeps spreading."
    ]
    
    for symptom in test_symptoms:
        prediction = assistant.diagnose(symptom)
        print(f"Symptoms: '{symptom}'")
        print(f"Predicted Diagnosis: -> **{prediction.upper()}**\n")