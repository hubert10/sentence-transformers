from sentence_transformers import SentenceTransformer, util

# Download model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Corpus of documents and their embeddings
corpus = [
    "Python is an interpreted high-level general-purpose programming language.",
    "Python is dynamically-typed and garbage-collected.",
    "The quick brown fox jumps over the lazy dog.",
]
corpus_embeddings = model.encode(corpus)

# Queries and their embeddings
queries = ["What is Python?", "What did the fox do?"]
queries_embeddings = model.encode(queries)

# Find the top-2 corpus documents matching each query
hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=2)

# Print results of first query
print(f"Query: {queries[0]}")
for hit in hits[0]:
    print(corpus[hit["corpus_id"]], "(Score: {:.4f})".format(hit["score"]))
# Query: What is Python?
# Python is an interpreted high-level general-purpose programming language. (Score: 0.6759)
# Python is dynamically-typed and garbage-collected. (Score: 0.6219)

# Print results of second query
print(f"Query: {queries[1]}")
for hit in hits[1]:
    print(corpus[hit["corpus_id"]], "(Score: {:.4f})".format(hit["score"]))
# Query: What did the fox do?
# The quick brown fox jumps over the lazy dog. (Score: 0.3816)
# Python is dynamically-typed and garbage-collected. (Score: 0.0713)
