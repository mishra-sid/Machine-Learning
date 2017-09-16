import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from time import time
from math import sqrt


class SVM(object):

    def __init__(self, s=0.8, q=5):
        self.fields = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
        self.field_type = {'age': 1, 'workclass': 0, 'fnlwgt': 1, 'education': 0, 'education-num': 1, 'marital-status': 0, 'occupation': 0, 'relationship': 0, 'race': 0, 'sex': 0, 'capital-gain': 1, 'capital-loss': 1, 'hours-per-week': 1, 'native-country': 0, 'class': 0}
        self.field_values = {
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
            'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th',
                          'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
            'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
            'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
                           'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
            'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
            'sex': ['Female', 'Male'],
            'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                               'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                               'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],

            'class': ['0', '1']
        }
        self.processed_data = {'train': [], 'test': []}
        self.labels = {'train': [], 'test': []}
        self.mx_dict = {}
        self.s = s
        self.q = q
        from csv import DictReader
        self.reader = DictReader

    def extract_and_process(self, filepath, file_type):
        # Extracting initial data from csv and begin processing
        with open(filepath, 'r') as tfile:
            curr_fields = self.fields
            if file_type == 'test':
                curr_fields = curr_fields[:-1]
            treader = self.reader(tfile, fieldnames=curr_fields)
            for row in treader:
                for key in row:
                    row[key] = row[key].strip()
                    if row[key] == '?':
                        continue
                    if self.field_type[key] == 0:
                        row[key] = self.field_values[key].index(row[key])
                    else:
                        row[key] = int(row[key])
                    if key == 'class':
                        continue
                    if file_type == 'train':
                        if key not in self.mx_dict:
                            self.mx_dict[key] = int(row[key])
                        else:
                            self.mx_dict[key] = max(self.mx_dict[key], int(row[key]))

                self.processed_data[file_type].append(row)

            # Processing question marks
            for key in self.fields[:-1]:
                avg = 0.0
                for row in self.processed_data[file_type]:
                    if row[key] == '?':
                        continue
                    avg += row[key]
                avg /= float(len(self.processed_data[file_type]))
                for row in self.processed_data[file_type]:
                    if row[key] == '?':
                        row[key] = avg

            # Normalize the vector elements
            for row in self.processed_data[file_type]:
                for key in row:
                    if key == 'class':
                        continue
                    row[key] /= float(self.mx_dict[key])

            if file_type == 'train':
                self.labels[file_type] = [d[self.fields[-1]] for d in self.processed_data[file_type]]
                self.processed_data[file_type] = [[d[x] for x in self.fields[:-1]] for d in self.processed_data[file_type]]
            else:
                self.processed_data[file_type] = [[d[x] for x in self.fields[:-1]] for d in self.processed_data[file_type]]

    def linearKernel(self, X, Y):
        return np.dot(X, Y.T)

    def polyKernel(self, X, Y):
        tp = np.dot(X, Y.T)
        tp += 1
        tp = np.power(tp, self.q)
        return tp

    def gaussKernel(self, X, Y):
        XY = 2 * np.dot(X, Y.T)
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)

        XXV = np.zeros((X.shape[0], 1))
        YYV = np.zeros((1, Y.shape[0]))

        for i in range(X.shape[0]):
            XXV[i][0] = XX[i][i]

        for i in range(Y.shape[0]):
            YYV[0][i] = YY[i][i]

        XXV = np.dot(XXV, np.ones((1, YYV.shape[0])))
        YYV = np.dot(np.ones((YYV.shape[0], 1)), YYV)

        return np.exp((XY - XXV - YYV) / (self.s * self.s))


class MultiKernelfixedrules(object):
    def __init__(self, kernels, X=None, Y=None):  # Initialize kernels, X and Y
        self.kernels = kernels
        self.X = X
        self.Y = Y

    def predict(self, a=0.25, b=0.60, c=0.15):  # predict using sklearn
        self.a = a
        self.b = b
        self.c = c
        class_multi = SVC(kernel=self.multi)
        class_multi.fit(self.X, self.Y)
        print 'predicted labels in Multi kernel fixed values', class_multi.predict(sv.processed_data['test'])

    def test_multi_kernel(self, a=0.25, b=0.60, c=0.15):  # Testing purposes
        self.a = a
        self.b = b
        self.c = c
        classif_multi = SVC(kernel=self.multi)
        for i in range(6):
            t_now = time()
            multi_scores = cross_val_score(classif_multi, self.X, self.Y, cv=5)
            t_next = time()
            print 'iteration=', (i + 1), 'accuracy=', multi_scores, 'time', t_next - t_now

    def multi(self, X, Y):  # multi kernel calls all kernels and does weighted sum
        return self.a * self.kernels[0](X, Y) + self.b * self.kernels[1](X, Y) + self.c * self.kernels[2](X, Y)


class MultiKernelheuristic(object):
    def __init__(self, kernels, X=None, Y=None):  # Initialize kernels, X and Y
        self.kernels = kernels
        self.X = X
        self.Y = Y
        self.nm = [0, 0, 0]
        self.gotindexes = 0

    def get_nm(self, X, Y):  # Get coefficients (a,b and c) using hueristic mentioned in question
        YY = np.dot(Y, Y.T)
        for i in range(len(self.kernels)):
            K = self.kernels[i](X, Y)
            fky = np.sum(np.multiply(K, YY))
            fkk = np.sum(np.multiply(K, K))
            self.nm[i] = fky / (len(self.X) * sqrt(fkk))
        S = sum(self.nm)
        self.nm = [x / S for x in self.nm]
        self.gotindexes = 1

    def predict(self):  # predict using sklearn
        class_multi = SVC(kernel=self.multi)
        class_multi.fit(self.X, self.Y)
        print 'predicted labels in MultiKernel heuristic', class_multi.predict(sv.processed_data['test'])

    def test_multi_kernel(self):
        classif_multi = SVC(kernel=self.multi)
        for i in range(6):
            t_now = time()
            multi_scores = cross_val_score(classif_multi, self.X, self.Y, cv=5)
            t_next = time()
            print 'iteration=', (i + 1), 'accuracy=', multi_scores, 'time', t_next - t_now

    def multi(self, X, Y):  # multi kernel calls all kernels and does weighted sum
        if not self.gotindexes:
            self.get_nm(X, Y)
            print self.nm
        return self.nm[0] * self.kernels[0](X, Y) + self.nm[1] * self.kernels[1](X, Y) + self.nm[2] * self.kernels[2](X, Y)


sv = SVM()
sv.extract_and_process('data/train.csv', 'train')
sv.extract_and_process('data/test.csv', 'test')
mk = MultiKernelfixedrules([sv.linearKernel, sv.polyKernel, sv.gaussKernel], sv.processed_data['train'], sv.labels['train'])
mk.predict()
# mk.test_multi_kernel()

mkh = MultiKernelheuristic([sv.linearKernel, sv.polyKernel, sv.gaussKernel], sv.processed_data['train'], sv.labels['train'])
# mkh.test_multi_kernel()
mkh.predict()
