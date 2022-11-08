from sentence_transformers import SentenceTransformer, util

# Download model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

# Look for paraphrases
paraphrases = util.paraphrase_mining(model, sentences)

# Print paraphrases
print("Top 5 paraphrases")
for paraphrase in paraphrases[0:5]:
    score, i, j = paraphrase
    print("Score {:.4f} ---- {} ---- {}".format(score, sentences[i], sentences[j]))
# Top 5 paraphrases
# Score 0.8939 ---- The new movie is awesome ---- The new movie is so great
# Score 0.6788 ---- The cat sits outside ---- The cat plays in the garden
# Score 0.5096 ---- I love pasta ---- Do you like pizza?
# Score 0.2560 ---- I love pasta ---- The new movie is so great
# Score 0.2440 ---- I love pasta ---- The new movie is awesome
