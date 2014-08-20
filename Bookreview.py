import wpf

from System.Windows import Window
from NBClassifier import *
from Preprocess import *

classifier_obj = NaiveBayesClassifer()
splitter = Splitter()

def create_classifier():
    #Create Naive Bayes Classifer and train it
    classifier = classifier_obj.get_sentiment_analysis_classifier()    
    classifier_obj.evaluate_classifier()

class Bookreview(Window):
    def __init__(self):
        wpf.LoadComponent(self, 'Bookreview.xaml')
    
    def Button_Click(self, sender, e):
        customer_review = self.tb_text.Text
        customer_words = splitter.split(customer_review)
        self.tb_Result.Text = classifier_obj.nb_classifier.classify(classifier_obj.bag_of_words(customer_words))
    
    def bt_Clear_Click(self, sender, e):
        self.tb_text.Text = ""
    
    def bt_Quit_Click(self, sender, e):
        self.Close()


     
