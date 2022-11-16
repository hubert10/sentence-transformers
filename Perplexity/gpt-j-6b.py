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


social_norms = [
    "People do not smile at strangers.",
    "Women cover their hair in church.",
    "Many women still expect their husband to be the breadwinner.",
    "People do not show their new baby to friends for a month or 40 days after they were born.",
    "Women frequently stay home with the child until the child is three years old/ women frequently don't return to work until their child is three years old.",
    "People will generally never eat or drink in public, apart from restaurants.",
    "People pass items to an older person with both hands",
    "Younger persons greet older persons first, and women greet men first",
    "Some young urbanites 'kiss the air' near each cheek while shaking hands.",
    "People shake right hands and may place the left hand under the right forearm as a sign of respect.",
    "The distance between people when they converse indicates their relationship: friends require little or no distance, while superiors must have more",
    "People toss their head to the side while uttering 'eh' to express disbelief, usually when they are listening to a personal experience",
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


####  TEST-1: on a new datases

# print(f"Incorrect sentence: ")
# print([score(i) for i in incorrect])
# print(f"Correct sentence: ")
# print([score(j) for j in correct])

#### TEST-2:  on social norms

print(f"Social norms: ")
print([score(j) for j in social_norms])
