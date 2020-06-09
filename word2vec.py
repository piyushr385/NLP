# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:57:52 2020

@author: piyush
"""

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = """"Dr Kalam's three step guide to achieve goals in life are: Finding an aim in life before you are twenty years old;
                Acquire knowledge continuously to reach this goal; Work hard and persevere so you can defeat all the problems
                and succeed.The challenge, my young friends, is that you have to fight the hardest battle, and ever stop fighting
                until you arrive at your destined place. What will be the tools with which you will fight this battle? 
                They are: have a great aim in life,continuously acquire the knowledge, work hard and persevere to realize the great achievement.
                We are as young as our faith and as old as our doubts. We are also as young as our self-confidence and as old as our fears.
                We are as young as our hopes and as old as our despairs.Creativity is seeing the same thing as everybody else, but thinking of something different.
                I would like to ask you, what would you like to be remembered for? You have to evolve yourself and shape your life. Write your dreams down on a piece of paper. 
                That page may be a very important page in the book of human history.Coming into contact with a good book and possessing it, is indeed an everlasting enrichment.
                Excellence happens not by accident. It is a process. You have to work hard to achieve it.
                The ignited minds of the youth is the most powerful resource on the Earth. I am convinced that the power of the youth, if properly directed, will bring about 
                transformed humanity by meeting its challenges and bring peace and rosperity.
                It doesn’t matter who you are, if you have a vision and determination to achieve that vision, you will certainly do so.
                Books become permanent companions. Sometimes, they are born before us; they guide us during our life journey and continue for many generations."""


# Preprocessing the data

text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)


# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)   #min_count:-Ignores all words with total frequency lower than this. depends on amount of data you are working with
words = model.wv.vocab   #vocabs found in the model 
#print(words)

# Most similar words
similar = model.wv.most_similar('destined')
print(similar)