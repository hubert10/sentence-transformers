# https://medium.com/nlplanet/two-minutes-nlp-sentence-transformers-cheat-sheet-2e9865083e7a

from sentence_transformers import SentenceTransformer, util
from PIL import Image

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Encode an image
img_emb = model.encode(Image.open("images/dog_on_table.jpeg"))

# Encode text descriptions
text_emb = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)

# Results:
# tensor([[0.1648, 0.2768, 0.1678]])
# The model predicts that the images matches with the 2nd
# description: 'A cat on a table'
