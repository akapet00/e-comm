import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from extract_data import HTML_CONTENT, extract_data


def main() -> None:
    """Print recommendations from the catalogue given user interests."""
    # set user interesets:
    user_interests = [
        "智能手机",  # smartphone
        "耳机",  # headphones
    ]
    print(f"User interests: {user_interests}", end="\n\n")

    # extract advertised products
    products = extract_data(HTML_CONTENT)
    product_names = [p["name"] for p in products]
    print(f"Set of advertised products: {product_names}", end="\n\n")

    # load an open-source Chinese Sentence BERT
    model = SentenceTransformer("uer/sbert-base-chinese-nli")

    # compute embeddings -> it could be a bit slow without a GPU
    embeddings = model.encode(product_names, show_progress_bar=True)

    # build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.asarray(embeddings))

    # now let's find the most similar item to the user interests
    query_embedding = model.encode(user_interests)
    _, i = index.search(query_embedding, 1)
    final_recs = {product_names[_i.item()] for _i in i.flatten()}
    print(f"Recommendations based on product-product similarity: {final_recs}")


if __name__ == "__main__":
    main()
