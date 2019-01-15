import nltk

nltk.download('treebank')
nltk.download('universal_tagset')

from nltk.corpus import treebank

tagged_sentences = treebank.tagged_sents(tagset='universal')

