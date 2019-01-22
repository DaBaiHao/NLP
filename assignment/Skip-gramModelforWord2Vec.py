import keras
import nltk
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import collections

austen = gutenberg.sents('austen-sense.txt') + gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt')

# This training corpus contains 16498 sentences. Use the line that follows to
# ensure that your code has the same number of lines.
print(len(austen))

# Pre-processing the Training Corpus
# remove special characters, empty strings, digits and stopwords from the sentences
# and put all the words into lower cases.
# normalized_corpus = []
'''
print(austen[0])

print(' '.join(austen[0]))
print(keras.preprocessing.text.text_to_word_sequence(' '.join(austen[0])))
'''
normalized_corpus = list()

for each_austen in austen:
    words = ' '.join(each_austen)

    if keras.preprocessing.text.text_to_word_sequence(words) != []:
        normalized_corpus.append(keras.preprocessing.text.text_to_word_sequence(words))


print('Length of processed corpus:', len(normalized_corpus))
print('Processed line:', normalized_corpus[10])