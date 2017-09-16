from DataSetManager import DataSetManager
from sklearn.neighbors import KNeighborsClassifier
from time import time


class KNearestNeighbours(object):

    def __init__(self, k): #initialize data 
        self.k = k
        self.data = DataSetManager({'train': 'train.csv', 'test': 'test.csv'},
                                   ['id', 'A', 'B', 'C', 'D', 'E', 'F', 'class'],
                                   [0, 100], [0, 1], 1000)
        self.data.generate()
        self.data.splitAndWrite(0.8)
        self.X = {}
        self.Y = {}

    def train(self): #train data
        self.data.load()
        self.predictions = []
        self.X['train'] = [[d[x] for x in self.data.fields[1:-1]] for d in self.data.training_set]
        self.Y['train'] = [d[self.data.fields[-1]] for d in self.data.training_set]
        self.X['test'] = [[d[x] for x in self.data.fields[1:-1]] for d in self.data.test_set]
        self.Y['test'] = [d[self.data.fields[-1]] for d in self.data.test_set]

    def predict(self):
        for nodex in self.data.test_set:
            distsqarr = []

            for nodey in self.data.training_set:
                distsq = 0
                for item, valuex in nodex.iteritems(): #get distances from all points in training space
                    if item is not 'id':
                        valuey = nodey[item]
                        distsq += (valuey - valuex) ** 2
                distsqarr.append({'class': nodey['class'], 'dist': distsq})

            distsqarr.sort(key=lambda node: node['dist']) #sort distances

            freq = {0: 0, 1: 0}
            for ind in range(min(len(self.data.training_set), self.k)): #take frequency in  k nearest and decide
                freq[distsqarr[ind]['class']] += 1

            if freq[1] > freq[0]:
                label = 1
            elif freq[1] == freq[0]:
                label = -1
            else:
                label = 0

            self.predictions.append(label)

    def test_accuracy(self):
        s = 0
        for curr in range(len(self.data.test_set)):
            s += (self.data.test_set[curr]['class'] == self.predictions[curr])
        return float(s) / len(self.data.test_set)

    def test_library(self):
        kxy = KNeighborsClassifier(n_neighbors=self.k)
        kxy.fit(self.X['train'], self.Y['train'])
        ll = kxy.predict(self.X['test'])
        s = 0
        for i in range(len(self.Y['test'])):
            if self.Y['test'][i] == ll[i]:
                s += 1
        return s / float(len(ll))

    def test_ks(self):
        for k in range(1, 22):
            self.k = k
            print 'k= ', k
            tx = time()
            self.train()
            self.predict()
            print self.test_accuracy()
            ty = time()
            print 'time cus', ty - tx
            tx = time()
            print self.test_library()
            ty = time()
            print 'time lib', ty - tx
            print


k = input()
knn = KNearestNeighbours(k)
knn.train()
knn.predict()
print knn.test_accuracy()
print knn.test_library()
