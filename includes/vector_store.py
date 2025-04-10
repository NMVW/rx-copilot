import numpy as np
import faiss
import os


def softmax(x):
    e_x = np.exp(x - np.max(x)) #subtract max for numerical stability.
    return e_x / e_x.sum()


class FaissIndex:

    """
    A class to handle the FAISS index for semantic search.
    It includes methods to create, save, remove, and load the index.
    """

    def __init__(self, faiss_index_path, dimension, overwrite=False):
        """Initializes the FaissIndex with a path to the FAISS index."""
        self.faiss_index_path = faiss_index_path
        self.index = None
        if overwrite:
            self._remove()
            self._create(dimension)
        else:
            self._load(dimension)
        print(f"FAISS index loaded from: {faiss_index_path}")

    def _remove(self):
        """Removes the FAISS index file."""
        if os.path.exists(self.faiss_index_path):
            os.remove(self.faiss_index_path)
            print(f"FAISS index removed from: {self.faiss_index_path}")
        else:
            print(f"Index not found at {self.faiss_index_path}. No action taken.")

    def _create(self, dimension):
        """Creates a FAISS index from the provided embeddings."""
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        self._save()
        print(f"FAISS index created with dimension: {dimension}")

    def _load(self, dimension):
        """Loads the FAISS index from the specified path."""
        if os.path.exists(self.faiss_index_path):
            try:
                self.index = faiss.read_index(self.faiss_index_path)
            except RuntimeError:
                print(f"Error loading index from {self.faiss_index_path}. Creating a new index.")
                self._create(dimension)
        else:
            print(f"Index not found at {self.faiss_index_path}. Creating a new index.")
            self._create(dimension)

    def _save(self):
        """Saves the FAISS index to the specified path."""
        faiss.write_index(self.index, self.faiss_index_path)
        print(f"FAISS index saved to: {self.faiss_index_path}")

    def add_embeddings(self, embeddings):
        """Adds new embeddings to the FAISS index."""
        for code, desc, embedding in embeddings:
            embedding_vector = embedding.astype('float32')
            self.index.add(embedding_vector.reshape(1, -1))
        self._save()

    def _search(self, query_embedding, top_k):
        """Searches the FAISS index for the top_k nearest neighbors."""
        similarities, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        return similarities, indices

    def get_prob_distribution(self, query_embedding, top_k):
        """
        Retrieves the top_k nearest neighbors in semantic space, and converts the distances to a probability distribution.
        """
        similarities, indices = self._search(query_embedding, top_k)
        probabilities = softmax(similarities[0])
        return probabilities, indices[0] # remove batch dimension.
