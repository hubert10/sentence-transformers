# -*- coding: utf-8 -*-
"""perplexity.ipynb
Author: Hubert K.

Original file is located at
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/tensorflow/perplexity.ipynb
"""

# Transformers installation
# ! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git

import torch
import sys
import numpy as np
 
from transformers import BloomTokenizerFast, BloomForCausalLM

model_name = "bigscience/bloom-560m"

# Load pre-trained model (weights)
with torch.no_grad():
        model = BloomForCausalLM.from_pretrained(model_name)
        model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
 
def score(sentence):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
 
social_norms = [
    "People do not smile at strangers.",
    "Women cover their hair in church.",
    "Women frequently stay home with the child until the child is three years old/ women frequently don't return to work until their child is three years old.",
    "Many women still expect their husband to be the breadwinner.",
    "People do not show their new baby to friends for a month or 40 days after they were born.",
]

incorrect = [
    "Our current population is 6 billion people and it is still growing exponentially.",
    "This will, if not already, caused  problems as there are very limited spaces for us.",
    "A manager should always be honest with their employees.",
    "They cooked the dinner themself.",
    "If I will be in London, I will contact to you."
]
correct = [
    "Our current population is 6 billion people, and it is still growing exponentially.",
    "This will, if not already, cause problems as there are very limited spaces for us.",
    "A manager should always be honest with his employees.",
    "They cooked the dinner themselves.",
    "If I am in London, I will contact you."
        
]

print(f"Incorrect sentence: ")
print([score(i) for i in incorrect])
print(f"Correct sentence: ")
print([score(i) for i in correct])
