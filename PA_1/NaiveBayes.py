from os import listdir
from math import sqrt, exp, pi


class NaiveBayes(object):

    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.spam_train_files = map(lambda x: 'spam-train/' + x, listdir('spam-train'))
        self.nonspam_train_files = map(lambda x: 'nonspam-train/' + x, listdir('nonspam-train'))
        self.spam_test_files = map(lambda x: 'spam-test/' + x, listdir('spam-test'))
        self.nonspam_test_files = map(lambda x: 'nonspam-test/' + x, listdir('nonspam-test'))

        import numpy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        self.np = numpy
        self.vectorizer = TfidfVectorizer()
        self.analyzer = self.vectorizer.build_analyzer()
        self.selector = SelectKBest(mutual_info_classif, k=50)

        self.training_vectors = {'spam': [], 'nonspam': []}
        self.testing_vectors = {'spam': [], 'nonspam': []}
        self.mean_class = {'spam': [0] * 50, 'nonspam': [0] * 50}
        self.var_class = {'spam': [0] * 50, 'nonspam': [0] * 50}
        self.true_positive_count = {'spam': 0, 'nonspam': 0}

    def extract(self):
        for file in self.spam_train_files:
            with open(file, 'r') as spam_file:
                content = self.analyzer(spam_file.read())
                matrix = self.vectorizer.fit_transform(content)
                input_vector = self.selector.fit_transform(matrix).toarray()
                self.training_vectors['spam'].append(input_vector)

        for file in self.nonspam_train_files:
            with open(file, 'r') as nonspam_file:
                content = self.analyzer(nonspam_file.read())
                input_vector = self.selector.fit_transform(self.vectorizer.fit_transform(content)).toarray()
                self.training_vectors['nonspam'].append(input_vector)

        for file in self.spam_test_files:
            with open(file, 'r') as spam_file:
                content = self.analyzer(spam_file.read())
                input_vector = self.selector.fit_transform(self.vectorizer.fit_transform(content)).toarray()
                self.testing_vectors['spam'].append(input_vector)

        for file in self.nonspam_test_files:
            with open(file, 'r') as nonspam_file:
                content = self.analyzer(nonspam_file.read())
                input_vector = self.selector.fit_transform(self.vectorizer.fit_transform(content)).toarray()
                self.testing_vectors['nonspam'].append(input_vector)

    def train(self):
        classwise_spamdata = zip(*self.training_vectors['spam'])
        for v in range(len(classwise_spamdata)):
            self.mean_class['spam'][v] = self.np.mean(classwise_spamdata[v])
            self.var_class['spam'][v] = self.np.var(classwise_spamdata[v])

        classwise_nonspamdata = zip(*self.training_vectors['nonspam'])
        for v in range(len(classwise_nonspamdata)):
            self.mean_class['nonspam'][v] = self.np.mean(classwise_nonspamdata[v])
            self.var_class['nonspam'][v] = self.np.var(classwise_nonspamdata[v])

    def get_likelihood(self, type_, class_, x):
        mean = self.mean_class[type_][class_]
        var = self.var_class[type_][class_]

        return (1.0 / (sqrt(2 * pi * var))) * exp(-((x - mean)**2) / (2 * var))

    def predict(self):
        for vector in self.testing_vectors['spam']:
            prob_spam, prob_nonspam = 1.0, 1.0
            for i in range(len(vector)):
                prob_spam *= self.get_likelihood('spam', i, vector[i])
                prob_nonspam *= self.get_likelihood('nonspam', i, vector[i])
            if prob_spam > prob_nonspam:
                self.true_positive_count['spam'] += 1

        for vector in self.testing_vectors['nonspam']:
            prob_spam, prob_nonspam = 1.0, 1.0
            for i in range(len(vector)):
                prob_spam *= self.get_likelihood('spam', i, vector[i])
                prob_nonspam *= self.get_likelihood('nonspam', i, vector[i])
            if prob_nonspam > prob_spam:
                self.true_positive_count['nonspam'] += 1

    def test_accuracy(self):
        print 'Accuracy in spam test data ', self.true_positive_count['spam'] / float(len(self.testing_vectors['spam']))
        print 'Accuracy in nonspam test data', self.true_positive_count['nonspam'] / float(len(self.testing_vectors['nonspam']))


nb = NaiveBayes()
nb.extract()
nb.train()
nb.predict()
nb.test_accuracy()
