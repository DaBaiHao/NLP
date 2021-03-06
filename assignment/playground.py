# import tree bank
import nltk

nltk.download('treebank')
nltk.download('universal_tagset')

from nltk.corpus import treebank

tagged_sentences = treebank.tagged_sents(tagset='universal')
'''
We now have the sentences tagged with universal tagset, the universal tagset has 12
    distinct tags:
        ● VERB - verbs (all tenses and modes)
        ● NOUN - nouns (common and proper)
        ● PRON - pronouns
        ● ADJ - adjectives
        ● ADV - adverbs
        ● ADP - adpositions (prepositions and postpositions)
        ● CONJ - conjunctions
        ● DET - determiners
        ● NUM - cardinal numbers
        ● PRT - particles or other function words
        ● X - other: foreign words, typos, abbreviations
        ● . - punctuation
'''

# The sentences are stored as lists of token, PoS tag pairs, we can take a look at the first
print(tagged_sentences[0])

#  prepare the data for building our model

print(len(tagged_sentences))

# assess the performance of our trained model
# we need to create a test set
train = tagged_sentences[:3000]
test = tagged_sentences[3000:]

# map the actual tokens and tags into integers.
# First let’s create a dictionary for tags
tagset = set([tag for sent in tagged_sentences for token,tag in sent])
tag2ids = {tag:id for id,tag in enumerate(tagset)}

print(tag2ids)

import collections

word_counter = collections.Counter([token.lower() for sent in train for token, tag in sent])
print(list(word_counter.items())[:10])



vocab = [k for k, v in word_counter.items() if v > 3]
word2ids = {token:id+2 for id, token in enumerate(vocab)}
word2ids['<UNK>'] = 0
word2ids['<PAD>'] = 1


print(list(word2ids.items())[:10])

import numpy as np

def preprocessing(data):
    labels = [tag2ids[tag] for sent in data for token, tag in sent]
    str_data = [['<PAD>' if i == 0 else sent[i - 1][0], sent[i][0], '<PAD>' if i == len(sent) - 1 else sent[i + 1][0]] for sent in data for i in range(len(sent))]
    out_data = [[word2ids[token] if token in word2ids else word2ids['<UNK>'] for token in
                 item] for item in str_data]
    return np.asarray(out_data), np.asarray(labels)

train_data,train_labels = preprocessing(train)
test_data, test_labels = preprocessing(test)
print(train_data[:10])
print(train_labels[:10])

import tensorflow as tf
from tensorflow import keras

vocab_size = len(word2ids.keys())
tag_size = len(tag2ids.keys())
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,100,input_shape=[3]))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.SimpleRNN(200,dropout=0.2,recurrent_dropout=0.2))
model.add(keras.layers.Dense(tag_size,activation=tf.nn.softmax))
model.summary()