# Linguistic Acceptability

# import libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)


sentence = "Luffy is a great pirate."
input_ids = tokenizer("cola: " + sentence, return_tensors="pt").input_ids
sentence_ids = model.generate(input_ids)
sentence = tokenizer.decode(sentence_ids[0], skip_special_tokens=True)
print(sentence)
