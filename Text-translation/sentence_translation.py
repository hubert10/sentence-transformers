# Task: Language Translation
# https://wandb.ai/mukilan/T5_transformer/reports/Exploring-Google-s-T5-Text-To-Text-Transformer-Model--VmlldzoyNjkzOTE2
# import libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)

language_sequence = "You should definitely watch 'One Piece', it is so good, you will love the comic book"
input_ids = tokenizer(
    "translate English to French: " + language_sequence, return_tensors="pt"
).input_ids
language_ids = model.generate(input_ids)
language_translation = tokenizer.decode(language_ids[0], skip_special_tokens=True)
print(language_translation)
