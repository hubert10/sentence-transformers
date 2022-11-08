# Sentence Similarity

# import libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)

stsb_sentence_1 = "Luffy was fighting in the war."
stsb_sentence_2 = "Luffy's fighting style is comical."
input_ids = tokenizer(
    "stsb sentence 1: " + stsb_sentence_1 + " sentence 2: " + stsb_sentence_2,
    return_tensors="pt",
).input_ids
stsb_ids = model.generate(input_ids)
stsb = tokenizer.decode(stsb_ids[0], skip_special_tokens=True)
print(stsb)
