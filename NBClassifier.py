import nltk
import yaml
import sys
import os
import re
from nltk.corpus import movie_reviews
import  collections
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


class NaiveBayesClassifer(object):
    def __init__(self):
        self.nb_classifier = None

    ''' it converts a list of words into a dictionary 
        Input: words - a list of words. [word1, word2, ...]
        Return a dictionary: {word1: True, word2: True, ... }
            It returns a bag of words because the words are not in order
            It just assign True to each word in the list '''
    def bag_of_words(self, words):
        return dict([(word, True) for word in words])

    ''' It filters out meaningless words in badwords
        Input: words, badwords  lists
        Return list of words without words in badwords '''
    def bag_of_words_not_in_set(self, words, badwords):
        return self.bag_of_words(set(words) - set(badwords))

    ''' stopwords : useless words, NLTK has a set of stopwords 
        It filters out stopwords
        Input: words list
        Return list without stopwords '''   
    def bag_of_non_stopwords(self, words):
        badwords = stopwords.words('english')
        return self.bag_of_words_not_in_set(words, badwords)

    " It adds the most common 200 bigram words to the bag of words"
    def  bag_of_bigrams_words(self, words, score_fn=BigramAssocMeasures.chi_sq, n=200):
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return self.bag_of_words(words + bigrams)

    '''
        Cropus: movie_reviews  
        Return a mapping of the form {label: [featureset]} 
        
        from nltk.corpus import movie_reviews
        from featx import label_feats_from_corpus, split_label_feats    
    '''

    '''  This mothod is for test only. It cut off the size of the corp for fast testing
    def label_feats_from_corpus(self, corp ) :  #feature_detector=bag_of_words):
         for label in corp.categories(): 
            for fileid in corp.fileids(categories=[label]):
                feats = self.bag_of_words(corp.words(fileids=[fileid]))  #feature_detector(corp.words(fileids=[fileid]))
                label_feats[label].append(feats)       
        for label in corp.categories():
            label_fileids = corp.fileids(categories=[label])
            label_fileids = label_fileids[:100]
            for fileid in label_fileids:
                feats = self.bag_of_words(corp.words(fileids=[fileid]))  
                label_feats[label].append(feats)
        return label_feats
    '''

    def label_feats_from_corpus(self, corp ) :  #feature_detector=bag_of_words):
         label_feats = collections.defaultdict(list)
         for label in corp.categories(): 
            for fileid in corp.fileids(categories=[label]):
                feats = self.bag_of_words(corp.words(fileids=[fileid]))  #feature_detector(corp.words(fileids=[fileid]))
                label_feats[label].append(feats)       
         return label_feats

    train_feats = []
    test_feats = []
    def split_label_feats(self, lfeats, split=0.75):      
        for label, feats in lfeats.iteritems():    
            cutoff = int(len(feats) * split)
            self.train_feats.extend([(feat, label) for feat in feats[:cutoff]])
            self.test_feats.extend([(feat, label) for feat in feats[cutoff:]])
        return self.train_feats, self.test_feats



    "Training the Naive Bayses Classifier "   
    def train_classifer(self, train_featuresets):
        self.nb_classifier = NaiveBayesClassifier.train(train_featuresets)
        return self.nb_classifier

    
    def get_sentiment_analysis_classifier(self, kind_classifier="NaiveBayses"):
        print kind_classifier
        if kind_classifier == "NaiveBayses" :
            if not self.nb_classifier :   # not null
                mv = movie_reviews
                print mv.categories()
                lfeats = self.label_feats_from_corpus(mv)
                training_feats, test_feats = self.split_label_feats(lfeats)
                self.train_classifer(training_feats)
            return self.nb_classifier                                      
        return

    def evaluate_classifier(self) :
        print accuracy(self.nb_classifier, self.test_feats)
        return



