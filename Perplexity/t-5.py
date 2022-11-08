from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

# Sample Sentences
correct = [
    "Humans have many basic needs and one of them is to have an environment that can sustain their lives"
    "Our current population is 6 billion people  and it is still growing exponentially.",
    "This will, if not already, caused  problems as there are very limited spaces for us.",
    "From large scale power generators to the basic cooking at  our homes, fuel is essential for all of these to happen and work.",
    # "In brief, innovators have to face many challenges when they want to develop the  products."	,
    # "The solution can be obtain by using technology to achieve a better usage of space that we have and resolve the problems in lands that  inhospitable  such as desserts  and swamps.",
    "As the number of people grows, the need of  habitable environment is unquestionably essential",
]
incorrect = [
    "Humans have many basic needs, and one of them is to have an environment that can sustain their lives.",
    "Our current population is 6 billion people, and it is still growing exponentially.",
    "This will, if not already, cause problems as there are very limited spaces for us.",
    "From large scale power generators to the basic cooking in our homes, fuel is essential for all of these to happen and work.",
    # "In brief, innovators have to face many challenges when they want to develop products.",
    # "The solution can be obtained by using technology to achieve a better usage of space that we have and resolve the problems in lands that are inhospitable, such as deserts and swamps.",
    "As the number of people grows, the need for a habitable environment is unquestionably essential.",
]

with torch.no_grad():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def score(sentence):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss = model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


####  TEST-1:  We can also test the model on a new datases

print(f"Incorrect sentence: ")
print([score(i) for i in incorrect])
print(f"Correct sentence: ")
print([score(i) for i in correct])

####  TEST-2:  We can also test the model on a new datases

a = [
    "there is a book on the desk",
    "there is a plane on the desk",
    "there is a book in the desk",
]
print([score(i) for i in a])
