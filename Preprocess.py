import nltk
import yaml
import sys
import os
import re
from nltk.corpus import movie_reviews
import  collections


''' This class Splitter splites a plain text to sentences first
'''
class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
    """
    def split(self, text):
       
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        #Temp convert it to a plain list 
        items = [item for a_list in tokenized_sentences for item in a_list ]
        #return tokenized_sentences
        return items





