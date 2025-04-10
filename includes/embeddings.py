import torch
from transformers import AutoTokenizer, AutoModel

class BioBERTEmbedder:

    dimension = 768 # BioBERT embedding dimension

    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        """Initializes the BioBERTEmbedder with a pre-trained model and tokenizer."""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print(f"Model {model_name} loaded successfully!")

        # Set the device to GPU, MPS, or CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        # Move the model to the selected device
        self.model = self.model.to(self.device)
        print(f"Model: {self.model}")

    def generate_embeddings_batch(self, texts):
        """Generates embeddings for a batch of texts."""
        # Cap token length for BERT at 512
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate mean of the hidden state
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        return embeddings.astype('float32')
