# Text Classification: Textual Entailment
# https://wandb.ai/mukilan/T5_transformer/reports/Exploring-Google-s-T5-Text-To-Text-Transformer-Model--VmlldzoyNjkzOTE2

# import libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)

entailment_premise = "I love One Piece."
entailment_hypothesis = "My feelings towards One Piece is filled with love"
input_ids = tokenizer(
    "mnli premise: " + entailment_premise + " hypothesis: " + entailment_hypothesis,
    return_tensors="pt",
).input_ids
entailment_ids = model.generate(input_ids)
entailment = tokenizer.decode(entailment_ids[0], skip_special_tokens=True)
print(entailment)
