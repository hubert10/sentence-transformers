# Sentence Transformers: Multilingual Sentence, Paragraph, and Image Embeddings using BLOOM, GPT-J-6B, BART and T5.

This repo provides examples on how to use LLMs to run most known NLP sentence tasks and how to compute the perplexity score on those tasks. Some of the large language models tested here **BLOOM**, **GPT-J-6B**, and **BART** and **T5**.

Each folder contains the code to test the corresponding tasks.

## Installation

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with pip**

Install the *sentence-transformers* with `pip`:

```
pip install -U sentence-transformers
```

**Install with conda**

You can install the *sentence-transformers* with `conda`:

```
conda install -c conda-forge sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documenation.

[This example](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/computing-embeddings/computing_embeddings.py) shows you how to use an already trained Sentence Transformer model to embed sentences for another task.

First download a pretrained model.

````python
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load pre-trained model (weights)
with torch.no_grad():
    model = BloomForCausalLM.from_pretrained(model_name)
    model.eval()

 # Load pre-trained model tokenizer (vocabulary)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)   
````
 
Then provide some sentences to the model.

````python
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
````
Then define a score function to compute the perplexity.

````python
def score(sentence):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss = model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
````

And that's it already. We now have a list of numpy arrays with labels.

````python
print(f"Social norms: ")
print([score(j) for j in social_norms])
````

## Pre-Trained Models

We provide a large list of [Pretrained Models](https://www.sbert.net/docs/pretrained_models.html) for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: `SentenceTransformer('model_name')`.

[»  Full list of pretrained models](https://www.sbert.net/docs/pretrained_models.html)

## Training

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

See [Training Overview](https://www.sbert.net/docs/training/overview.html) for an introduction how to train your own embedding models. We provide [various examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training) how to train models on various datasets.

Some highlights are:
- Support of various transformer networks including BLOOM, GPT-J-6B, T5, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [10+ loss-functions](https://www.sbert.net/docs/package_reference/losses.html) allowing to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss.

## Performance

Our models are evaluated extensively on 15+ datasets including challening domains like Tweets, Reddit, emails. They achieve by far the **best performance** from all available sentence embedding methods. Further, we provide several **smaller models** that are **optimized for speed**.

## Incorrect sentences

The models are evaluated on the Preplexity score: the lower score on correct sentences and high score on incorrect sentences indicates good performance model:

| Models | BLOOM | GPT-J-6B |BART | T5 |
| ---- |:----:|:----:|:----:|:----:|
| Our current population is 6 billion people and it is still growing exponentially. | 29.490658 | 18.9348 | 1.0000235 | 3.860105 |
| This will, if not already, caused  problems as there are very limited spaces for us. | 103.92918 | 111.370026 | 1.0057222 | 2.565533 |
| A manager should always be honest with their employees. | 34.297775 | 28.413712 | 1.0000546 | 2.4984558 |
| They cooked the dinner themself. | 329.0331 | 167.6923 | 1.0001534 | 13.014829 |
| If I will be in London, I will contact to you. |  45.87319 | 18.13029 | 1.0001565 | 3.2314863 |

#### Correct sentences

| Models | BLOOM | GPT-J-6B |BART | T5 |
| ---- |:----:|:----:|:----:|:----:|
| Our current population is 6 billion people and it is still growing exponentially. | 23.996367 | 16.459309 | 1.0000113 | 3.28777 |
| This will, if not already, caused  problems as there are very limited spaces for us. | 45.060555 | 51.95318 | 1.0000496 | 2.5356748 |
| A manager should always be honest with their employees. | 29.250494 | 26.321022 | 1.000057 | 2.493631 |
| They cooked the dinner themself. | 232.12733 | 102.1904 | 1.0007304 | 9.535956 |
| If I will be in London, I will contact to you. | 24.296467 | 20.484407 | 1.0000682 | 4.7148967 |

## Application Examples

You can use this repo for:

- [Computing Sentence Embeddings](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
- [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Clustering](https://www.sbert.net/examples/applications/clustering/README.html)
- [Paraphrase Mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)
 - [Translated Sentence Mining](https://www.sbert.net/examples/applications/parallel-sentence-mining/README.html)
 - [Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html)
 - [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) 
 - [Text Summarization](https://www.sbert.net/examples/applications/text-summarization/README.html) 
- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/applications/image-search/README.html)

and many more use-cases.



