from sentence_transformers import SentenceTransformer

# Download model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# The sentences we'd like to encode
sentences = [
    "Python is an interpreted high-level general-purpose programming language.",
    "Python is dynamically-typed and garbage-collected.",
    "The quick brown fox jumps over the lazy dog.",
]

# Get embeddings of sentences
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
# Sentence: Python is an interpreted high-level general-purpose programming language.
# Embedding: [-1.17965914e-01 -4.57159936e-01 -5.87313235e-01 -2.72477478e-01 ...
