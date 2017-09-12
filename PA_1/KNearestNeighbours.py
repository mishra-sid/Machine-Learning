from DataSetManager import DataSetManager


class KNearestNeighbours(object):
    """docstring for KNearestNeighbours"""

    def __init__(self, k):
        super(KNearestNeighbours, self).__init__()
        self.k = k
        self.data = DataSetManager({'train': 'train.csv', 'test': 'test.csv'},
                                   ['id', 'A', 'B', 'C', 'D', 'E', 'F', 'class'],
                                   [0, 100], [0, 1], 2000)
        self.data.generate()
        self.data.splitAndWrite(0.8)

    def train(self):
        self.data.load()
        self.predictions = []

    def predict(self):
        for nodex in self.data.test_set:
            distsqarr = []

            for nodey in self.data.training_set:
                distsq = 0
                for item, valuex in nodex.iteritems():
                    valuey = nodey[item]
                    distsq += abs(valuey - valuex) ** 2
                distsqarr.append({'class': nodey['class'], 'dist': distsq})

            distsqarr.sort(key=lambda node: node['dist'])

            freq = {0: 0, 1: 0}
            for ind in range(min(len(self.data.training_set), self.k)):
                freq[distsqarr[ind]['class']] += 1

            label = 0
            if freq[1] > freq[0]:
                label = 1
            self.predictions.append(label)

    def test_accuracy(self):
        s = 0
        for curr in range(len(self.data.test_set)):
            s += (self.data.test_set[curr]['class'] == self.predictions[curr])
        return float(s) / len(self.data.test_set)


k = 12
knn = KNearestNeighbours(k)
knn.train()
knn.predict()
print knn.test_accuracy()
