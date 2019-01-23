from tensorflow import keras
import nltk
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from nltk.corpus import stopwords


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

def pre_processing_training_data(tokenozed_paragraph):
    # init the output array
    normalized_corpus = list()

    # download stopwords
    nltk.download('stopwords')
    stopWords = set(stopwords.words('english'))

    # print(stopWords)
    for each_austen in tokenozed_paragraph:
        # remove punctuation
        words = ' '.join(each_austen)
        words = keras.preprocessing.text.text_to_word_sequence(words)

        # remove stop words
        for each_word in words:
            if each_word in stopWords:
                words.remove(each_word)

        if words != []:
            normalized_corpus.append(words)
    return normalized_corpus
'''
normalized_corpus = list()

# download stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
# print(stopWords)
for each_austen in austen:
    # remove punctuation
    words = ' '.join(each_austen)
    words = keras.preprocessing.text.text_to_word_sequence(words)

    # remove stop words
    for each_word in words:
        if each_word in stopWords:
            words.remove(each_word)

    if words != []:
        normalized_corpus.append(words)
'''
normalized_corpus = pre_processing_training_data(austen)
print('Length of processed corpus:', len(normalized_corpus))
print('Processed line:', normalized_corpus[10])


word2idx = dict()
idx2word = list()
corpusVocabulary = list()
for each_words in normalized_corpus:
    for each_word in each_words:
        if each_word not in word2idx:
            word2idx[each_word] = 1
            idx2word.append(each_word)
        else:
            word2idx[each_word] += 1


sents_as_ids = list()
count_num = 0
for each_words in normalized_corpus:
    sents_as_ids.append([])
    for each_word in each_words:
        index = idx2word.index(each_word)
        sents_as_ids[count_num].append(index)
    count_num += 1

print('\nSample word2idx: ', list(word2idx.items())[:10])

print(len(word2idx))
print(len(idx2word))
print('\nSample word2idx: ', list(word2idx.items())[:10])
print('\nSample normalized corpus:', normalized_corpus[:3])
print('\nAbove sentence as a list of ids:' , sents_as_ids[:3])
print(len(sents_as_ids))


# training
from keras.preprocessing.sequence import skipgrams
skip_grams = [skipgrams(sent, vocabulary_size=vocab_size, window_size=5) for sent in sents_as_ids]