# -*- coding: utf-8 -*-
"""T5_Transformer_Tasks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fC5VpXC872E6vh_VjLRu5o6bo7lmIOJM

# T5_Transformer tasks

# 0. Importing Libraries
"""

!pip install transformers

! pip install sentencepiece

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

"""# 1. Text Summarization"""

one_piece_sequence = ("The series focuses on Monkey D. Luffy, a young man made of rubber, who, inspired by his childhood idol," 
             "the powerful pirate Red-Haired Shanks, sets off on a journey from the East Blue Sea to find the mythical treasure," 
             "the One Piece, and proclaim himself the King of the Pirates. In an effort to organize his own crew, the Straw Hat Pirates," 
             "Luffy rescues and befriends a pirate hunter and swordsman named Roronoa Zoro, and they head off in search of the " 
             "titular treasure. They are joined in their journey by Nami, a money-obsessed thief and navigator; Usopp, a sniper "
             "and compulsive liar; and Sanji, a perverted but chivalrous cook. They acquire a ship, the Going Merry, and engage in confrontations"  
             "with notorious pirates of the East Blue. As Luffy and his crew set out on their adventures, others join the crew later in the series, "
             "including Tony Tony Chopper, an anthropomorphized reindeer doctor; Nico Robin, an archaeologist and former Baroque Works assassin; "
             "Franky, a cyborg shipwright; Brook, a skeleton musician and swordsman; and Jimbei, a fish-man helmsman and former member of the Seven "
             "Warlords of the Sea. Once the Going Merry is damaged beyond repair, Franky builds the Straw Hat Pirates a new ship, the Thousand Sunny," 
             "Together, they encounter other pirates, bounty hunters, criminal organizations, revolutionaries, secret agents, and soldiers of the" 
             "corrupt World Government, and various other friends and foes, as they sail the seas in pursuit of their dreams.")

inputs = tokenizer.encode("summarize: " + one_piece_sequence,
                          return_tensors='pt',
                          max_length=512,
                          truncation=True)

summarization_ids = model.generate(inputs, max_length=80, min_length=40, length_penalty=5., num_beams=2)

summarization = tokenizer.decode(summarization_ids[0])

summarization

"""# 2. Language Translation"""

language_sequence = ("You should definitely watch 'One Piece', it is so good, you will love the comic book")

input_ids = tokenizer("translate English to French: "+language_sequence, return_tensors="pt").input_ids

language_ids = model.generate(input_ids)

language_translation = tokenizer.decode(language_ids[0],skip_special_tokens=True)

language_translation

"""# 3. Text Classification: Textual Entailment"""

entailment_premise = ("I love One Piece.")
entailment_hypothesis = ("My feelings towards One Piece is filled with love")

input_ids = tokenizer("mnli premise: "+entailment_premise+" hypothesis: "+entailment_hypothesis, return_tensors="pt").input_ids

entailment_ids = model.generate(input_ids)

entailment = tokenizer.decode(entailment_ids[0],skip_special_tokens=True)

entailment

"""# 4. Linguistic Acceptability """

sentence = ("Luffy is a great pirate.")

input_ids = tokenizer("cola: "+ sentence, return_tensors="pt").input_ids

sentence_ids = model.generate(input_ids)

sentence = tokenizer.decode(sentence_ids[0],skip_special_tokens=True)

sentence

"""# 5. Sentence Similarity"""

stsb_sentence_1 = ("Luffy was fighting in the war.")
stsb_sentence_2 = ("Luffy's fighting style is comical.")

input_ids = tokenizer("stsb sentence 1: "+stsb_sentence_1+" sentence 2: "+stsb_sentence_2, return_tensors="pt").input_ids

stsb_ids = model.generate(input_ids)

stsb = tokenizer.decode(stsb_ids[0],skip_special_tokens=True)

stsb