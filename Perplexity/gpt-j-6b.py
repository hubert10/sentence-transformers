import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# Sample Sentences
incorrect = [
    "Our current population is 6 billion people and it is still growing exponentially.",
    "This will, if not already, caused  problems as there are very limited spaces for us.",
    "A manager should always be honest with their employees.",
    "They cooked the dinner themself.",
    "If I will be in London, I will contact to you.",
]
correct = [
    "Our current population is 6 billion people, and it is still growing exponentially.",
    "This will, if not already, cause problems as there are very limited spaces for us.",
    "A manager should always be honest with his employees.",
    "They cooked the dinner themselves.",
    "If I am in London, I will contact you.",
]


# https://betterprogramming.pub/fine-tuning-gpt-j-6b-on-google-colab-or-equivalent-desktop-or-server-gpu-b6dc849cb205
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


def score(sentence):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss = model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


####  TEST-1:  We can also test the model on a new datases

print(f"Incorrect sentence: ")
print([score(i) for i in incorrect])
print(f"Correct sentence: ")
print([score(j) for j in correct])
# print(i for i in correct)

#### TEST-2:  We can also test the model on a new datases

# a = [
#     'there is a book on the desk',
#     'there is a plane on the desk',
#     'there is a book in the desk'
#   ]
# print([score(i) for i in a])
