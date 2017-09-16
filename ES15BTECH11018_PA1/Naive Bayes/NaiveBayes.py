from os import listdir
from math import sqrt, pi, log
from time import time


class NaiveBayes(object):

    def __init__(self): #Initialize data
        self.spam_train_files = map(lambda x: 'spam-train/' + x, listdir('spam-train')) #Get filename list
        self.nonspam_train_files = map(lambda x: 'nonspam-train/' + x, listdir('nonspam-train'))
        self.spam_test_files = map(lambda x: 'spam-test/' + x, listdir('spam-test'))
        self.nonspam_test_files = map(lambda x: 'nonspam-test/' + x, listdir('nonspam-test'))

        import numpy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        self.np = numpy
        self.vectorizer = TfidfVectorizer()
        self.selector = SelectKBest(mutual_info_classif, k=50) #Import and initialize things from sklearn

        self.training_vectors = {'spam': [], 'nonspam': []}
        self.testing_vectors = {'spam': [], 'nonspam': []}
        self.mean_class = {'spam': [0] * 50, 'nonspam': [0] * 50}
        self.var_class = {'spam': [0] * 50, 'nonspam': [0] * 50}
        self.true_positive_count = {'spam': 0, 'nonspam': 0}

    def extract(self):
        content = []
        for file in self.spam_train_files:
            with open(file, 'r') as spam_file:
                content.append(spam_file.read())
        for file in self.nonspam_train_files:
            with open(file, 'r') as nonspam_file:
                content.append(nonspam_file.read()) #get content from text files

        matrix = self.vectorizer.fit_transform(content)
        input_vectors = self.selector.fit_transform(matrix, [0] * len(self.spam_train_files) + [1] * len(self.nonspam_train_files)).toarray()

        self.X = input_vectors

        self.training_vectors['spam'].extend(input_vectors[: len(self.spam_train_files)]) #Convert to martrix and split
        self.training_vectors['nonspam'].extend(input_vectors[len(self.spam_train_files):])

        content = []
        for file in self.spam_test_files:
            with open(file, 'r') as spam_file:
                content.append(spam_file.read()) #get content from text files

        mat = self.vectorizer.transform(content)
        input_vectors = self.selector.transform(mat).toarray()

        self.testing_vectors['spam'].extend(input_vectors)
        content = []

        for file in self.nonspam_test_files:
            with open(file, 'r') as nonspam_file:
                content.append(nonspam_file.read())#get content from text files

        mat = self.vectorizer.transform(content)
        input_vectors = self.selector.transform(mat).toarray()
        self.testing_vectors['nonspam'].extend(input_vectors)

    def train(self): #training is simply calculate mean and variance of each attr 
        classwise_spamdata = zip(*self.training_vectors['spam'])
        for v in range(len(classwise_spamdata)):
            self.mean_class['spam'][v] = self.np.mean(classwise_spamdata[v])
            self.var_class['spam'][v] = self.np.var(classwise_spamdata[v])

        classwise_nonspamdata = zip(*self.training_vectors['nonspam'])
        for v in range(len(classwise_nonspamdata)):
            self.mean_class['nonspam'][v] = self.np.mean(classwise_nonspamdata[v])
            self.var_class['nonspam'][v] = self.np.var(classwise_nonspamdata[v])

    def get_likelihood(self, type_, class_, x): #get likelihood P(X|Y) of each attr
        mean = self.mean_class[type_][class_]
        var = self.var_class[type_][class_]
        if var == 0.0:
            return 0.0
        return log(1.0 / (sqrt(2 * pi * var))) - ((x - mean)**2) / (2 * var)

    def predict(self): 
        for vector in self.testing_vectors['spam']:
            prob_spam, prob_nonspam = 0.0, 0.0
            for i in range(len(vector)):
                prob_spam += self.get_likelihood('spam', i, vector[i])
                prob_nonspam += self.get_likelihood('nonspam', i, vector[i])
            if prob_spam > prob_nonspam: #True positive prediction
                self.true_positive_count['spam'] += 1

        for vector in self.testing_vectors['nonspam']:
            prob_spam, prob_nonspam = 0.0, 0.0
            for i in range(len(vector)):
                prob_spam += self.get_likelihood('spam', i, vector[i])
                prob_nonspam += self.get_likelihood('nonspam', i, vector[i])
            if prob_nonspam > prob_spam: #True negative prediction
                self.true_positive_count['nonspam'] += 1

    def test_accuracy(self):
        print 'Accuracy in spam test data ', self.true_positive_count['spam'] / float(len(self.testing_vectors['spam']))
        print 'Accuracy in nonspam test data', self.true_positive_count['nonspam'] / float(len(self.testing_vectors['nonspam']))

    def test_library(self):
        from sklearn.naive_bayes import GaussianNB
        gb = GaussianNB()
        gb.fit(self.X, [0] * len(self.spam_train_files) + [1] * len(self.nonspam_train_files))
        g1, g2 = list(gb.predict(self.testing_vectors['spam'])), list(gb.predict(self.testing_vectors['nonspam']))
        print g1.count(0) / float(len(g1))
        print g2.count(1) / float(len(g2))


nb = NaiveBayes()
nb.extract()
nb.train()
nb.predict()
nb.test_accuracy()
nb.test_library()
