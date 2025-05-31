import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extractor import HTML_CONTENT, extract_data


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

    # fit a character-level vectorizer (as Chinese needs special tokenization)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
    vecs = vectorizer.fit_transform(product_names + user_interests)
    prod_vecs = vecs[: len(product_names)]
    user_vecs = vecs[len(product_names) :]
    sim_matrix = cosine_similarity(prod_vecs, user_vecs)

    # for each product, check if it matches any interest above threshold
    threshold = 0.1
    final_recs = []
    for i, prod in enumerate(products):
        if np.max(sim_matrix[i]) > threshold:
            final_recs.append(prod)
    print(f"Recommendations based on product-product similarity: {final_recs}")


if __name__ == "__main__":
    main()
