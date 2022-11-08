import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch

model = BloomForCausalLM.from_pretrained("bigscience/bloom")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
# Speaking of which, let’s set some globals, including our prompt text:

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

# A few notes:

# result_length calibrates the size of the response (in tokens) we get for the prompt from the model.
# inputs contains the embedding representation of prompt, encoded for use specifically by PyTorch. If we were using TensorFlow we’d pass return_tensors="tf".
# Running Inference: Strategies for Better Responses
# Before we send the model our prompt, we need to think about which decoding / search strategies might work best for our use case. With autoregressive transformers (trained for next token prediction) we have a number of options to search the answer space for the most “reasonable” output. This great article by Patrick von Platen (Huggingface) does an excellent job explaining the details and math behind the 3 techniques we’ll be trying, so I won’t reinvent the wheel here. I will however, give you the TL;DR version of each:

# Greedy Search simply chooses the next word at each timestep t+1 that has the highest predicted probability of following the word at t. One of the main issues here is that greedy search will miss words with a high probability at t+1 if it is preceded by a word with a low probability at t.
# Beam Search keeps track of the n-th (num_beams) most likely word sequences and outputs the most likely sequence. Sounds great, but this method breaks down when the output length can be highly variable — as in the case of open-ended text generation. Both greedy and beam search also produce outputs whose distribution does not align very well with the way humans might perform the same task (i.e. both are liable to produce fairly repetitive, boring text).
# Sampling With Top-k + Top-p is a combination of three methods. By sampling, we mean that the next word is chosen randomly based on its conditional probability distribution (von Platen, 2020). In Top-k, we choose the k most likely words, and then redistribute the probability mass amongst them before the next draw. Top-p adds an additional constraint to top-k, in that we’re choosing from the smallest set of words whose cumulative probability exceed p.
# Now we’ll try all 3 strategies so we can compare the outputs.:

# Greedy Search
print(tokenizer.decode(model.generate(inputs["input_ids"], 
                       max_length=result_length
                      )[0]))
'''It was a dark and stormy night, and the wind was blowing hard. The
snow was falling fast, and the ground was covered with it. The
horses were all frozen to the ground, and the men were huddled'''


# Beam Search
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length, 
                       num_beams=2, 
                       no_repeat_ngram_size=2,
                       early_stopping=True
                      )[0]))
'''It was a dark and stormy night, and the wind was blowing hard. I was in the
middle of the road, when I heard a loud crash. It came from the house
at the other side of my road. A man was'''

# Sampling Top-k + Top-p
print(tokenizer.decode(model.generate(inputs["input_ids"],
                       max_length=result_length, 
                       do_sample=True, 
                       top_k=50, 
                       top_p=0.9
                      )[0]))
'''It was a dark and stormy night. It was almost noon. As I got out of the car and took off my shoes, a man walked over to me and sat down. He had a mustache, thick hair and brown eyes. He'''

