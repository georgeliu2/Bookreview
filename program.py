import wpf
from System.Windows import Application, Window
import sys
sys.path.append(r'C:\Python27\Lib')
sys.path.append(r'C:\Python27\Lib\site-packages')
import clr
clr.AddReference("System.Windows.Forms")

from Bookreview import *
from NBClassifier import *

    

if __name__ == '__main__':
    #Create Naive Bayes Classifer and train it
    create_classifier()
    Application().Run(Bookreview())
