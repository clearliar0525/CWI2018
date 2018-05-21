import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import tree,svm
from nltk.corpus import wordnet as wn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from collections import Counter
import nltk
import re
import numpy as np

class Baseline(object):

    def __init__(self, language,option):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
        if option=="RandomForest":
            self.model = RandomForestClassifier()
        if option=="svm":
            self.model = svm.SVC()
        if option=="tree":
            self.model=tree.DecisionTreeClassifier()
        if option=="LogisticRegression":
            self.model=LogisticRegression()
        if option=="GradientBoosting":
            self.model=GradientBoostingClassifier()
        

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        if self.language=='english':
            syno=len(wn.synsets(re.sub("[^\w]"," ",word)))
        else:
             syno=len(wn.synsets(word[0]))
        return [len_chars,len_tokens,syno,word]
    
        
    def freqdict(self, dataset):
        wordnum=[]
        for sent in dataset:
            wordnum.append(self.extract_features(sent['target_word'])[3])
        x2=Counter(wordnum)
        return x2
        
    def posdict(self,dataset):
        posnum=[]
        for sent in dataset:
            posnum.append(nltk.pos_tag(self.extract_features(sent['target_word'])[3])[0][1]) 
        setposnum = set(posnum)    
        posindex = {word: i for i, word in enumerate(setposnum)}
        return posindex

    def train(self,trainset,freqdict1,posindex1):
        X = []
        y = []
        for sent in trainset:
            word=self.extract_features(sent['target_word'])[3]
            wordpos=nltk.pos_tag(word)
            X.append(np.hstack((self.extract_features(sent['target_word'])[:3],posindex1[wordpos[0][1]],freqdict1[word])))
            y.append(sent['gold_label'])
        self.model.fit(X, y)
        plt.figure()
        plt.title("Learning Curves of random forest")
        plt.xlabel("Training samples")
        plt.ylabel("Score")
        estimator = self.model
        train_sizes = np.linspace(0.1, 1.0,10)
        train_sizes, train_scores,test_scores= learning_curve(estimator, X, y, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
        plt.show()
    

    def test(self,testset,freqdict1,posindex1):
        X = []
        for sent in testset:
            word=self.extract_features(sent['target_word'])[3]
            wordpos=nltk.pos_tag(word)
            X.append(np.hstack((self.extract_features(sent['target_word'])[:3],posindex1[wordpos[0][1]],freqdict1[word])))

        return self.model.predict(X)

