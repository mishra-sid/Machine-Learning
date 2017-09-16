import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from time import time


class SVM(object): # Initialize data

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

            #Get matrix from dictionary
            if file_type == 'train':
                self.labels[file_type] = [d[self.fields[-1]] for d in self.processed_data[file_type]]
                self.processed_data[file_type] = [[d[x] for x in self.fields[:-1]] for d in self.processed_data[file_type]]
            else:
                self.processed_data[file_type] = [[d[x] for x in self.fields[:-1]] for d in self.processed_data[file_type]]

    def linearKernel(self, X, Y): #Linear Kernel is simple dot product 
        return np.dot(X, Y.T)

    def polyKernel(self, X, Y): #Polynomial Kernel is (X.yT + 1)^d 
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

        for i in range(X.shape[0]): #Get the required columns from XX and YY
            XXV[i][0] = XX[i][i]

        for i in range(Y.shape[0]):
            YYV[0][i] = YY[i][i]

        XXV = np.dot(XXV, np.ones((1, YYV.shape[0]))) #Multiply with appropriate np.ones to make it of same dimension as XY
        YYV = np.dot(np.ones((YYV.shape[0], 1)), YYV)

        return np.exp((XY - XXV - YYV) / (self.s * self.s)) #Obtain final result

    def predict(self): #Use inbuilt libraries to predict
        class_linear = SVC(kernel=self.linearKernel)
        class_linear.fit(self.processed_data['train'], self.labels['train'])

        print 'predicted labels in Linear kernel', class_linear.predict(self.processed_data['test'])

        class_poly = SVC(kernel=self.polyKernel)
        class_poly.fit(self.processed_data['train'], self.labels['train'])
        print 'predicted labels in Polynomical kernel', class_poly.predict(self.processed_data['test'])

        class_gaussian = SVC(kernel=self.gaussKernel)
        class_gaussian.fit(self.processed_data['train'], self.labels['train'])
        print 'predicted labels in Gaussian Kernel', class_gaussian.predict(self.processed_data['test'])

    def test_kernels(self):
        for i in range(5):
            t_now = time()
            class_linear = SVC(kernel=self.linearKernel)
            linear_scores = cross_val_score(class_linear, self.processed_data['train'], self.labels['train'], cv=5)
            t_next = time()
            print 'iteration=', (i + 1), 'accuracy=', linear_scores, 'time=', t_next - t_now

        for i in range(6):
            t_now = time()
            self.q = i + 1
            class_poly = SVC(kernel=self.polyKernel)
            poly_scores = cross_val_score(class_poly, self.processed_data['train'], self.labels['train'], cv=5)
            t_next = time()
            print 'q =', self.q, 'accuracy=', poly_scores, 'time=', t_next - t_now

        for i in range(9):
            t_now = time()
            self.s = 0.1 * (i + 1)
            class_gaussian = SVC(kernel=self.gaussKernel)
            gaussian_scores = cross_val_score(class_gaussian, self.processed_data['train'], self.labels['train'], cv=5)
            t_next = time()
            print 's=', self.s, 'accuracy=', gaussian_scores, 'time=', t_next - t_now


sv = SVM()
sv.extract_and_process('data/train.csv', 'train')
#sv.test_kernels()
sv.extract_and_process('data/test.csv', 'test')
sv.predict()
