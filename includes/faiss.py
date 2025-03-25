import numpy as np
import faiss


def softmax(x):
    e_x = np.exp(x - np.max(x)) #subtract max for numerical stability.
    return e_x / e_x.sum()


def get_probability_distribution(query_embedding, index, top_k):
    """
    Retrieves the top_k nearest neighbors in semantic space, and converts the distances to a probability distribution.
    """
    similarities, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
    probabilities = softmax(similarities[0])
    return probabilities, indices[0] # remove batch dimension.


def build_faiss_index(embeddings, faiss_index_path):
    """Builds and saves a FAISS index with BioBERT embedding dimension."""
    dimension = 768  # BioBERT embedding dimension
    index = faiss.IndexFlatIP(dimension)

    for code, desc, embedding in embeddings:
        embedding_vector = embedding.astype('float32')
        index.add(embedding_vector.reshape(1, -1))

    # Save the index
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index saved to: {faiss_index_path}")

    return index