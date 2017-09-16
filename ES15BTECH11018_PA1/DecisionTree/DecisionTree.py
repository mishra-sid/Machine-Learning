import pickle


class DecisionTree(object):

    def __init__(self, bin_size=5):  # Initialize required fields
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

            'class': [0, 1]
        }
        self.bin_size = bin_size
        for key in self.field_type:
            if self.field_type[key] == 1:
                self.field_values[key] = list(range(1, self.bin_size + 1))
        self.processed_data = {'train': [], 'test': []}
        self.bin_dict = {}
        self.tree = {}
        self.predictions = []

        from math import ceil, log
        from csv import DictReader
        self.log = log
        self.ceil = ceil
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
                    if key == 'class':
                        row[key] = int(row[key])
                        continue

                    if self.field_type[key] == 1:
                        if key not in self.bin_dict:
                            self.bin_dict[key] = {'min': int(row[key]), 'max': int(row[key])}
                        else:
                            self.bin_dict[key]['min'] = min(self.bin_dict[key]['min'], int(row[key]))
                            self.bin_dict[key]['max'] = max(self.bin_dict[key]['max'], int(row[key]))
                        row[key] = int(row[key])

                self.processed_data[file_type].append(row)

            # Binning for continuous data
            for row in self.processed_data[file_type]:
                for key in row:
                    if row[key] == '?':
                        continue
                    if self.field_type[key] == 1:
                        if self.bin_dict[key]['min'] == self.bin_dict[key]['max']:
                            bin_id = 1
                        else:
                            bin_id = self.ceil((float(row[key] - self.bin_dict[key]['min']) * self.bin_size) / float(self.bin_dict[key]['max'] - self.bin_dict[key]['min']))
                            if bin_id == int(bin_id) and bin_id != self.bin_size:
                                bin_id = int(bin_id) + 1
                            else:
                                bin_id = int(bin_id)
                        row[key] = bin_id

            # Processing question marks
            for key in self.fields[:-1]:
                freq = {}
                for val in self.field_values[key]:
                    freq[val] = 0.0
                for row in self.processed_data[file_type]:
                    if row[key] == '?':
                        continue
                    freq[row[key]] += 1.0
                best_key = max(freq, key=freq.get)
                for row in self.processed_data[file_type]:
                    if row[key] == '?':
                        row[key] = best_key

    def best_attr_to_split(self, curr_data):
        least_after_split_entropy = -1
        for key in self.fields[:-1]:
            after_split_dict = {}
            for val in self.field_values[key]:
                after_split_dict[val] = {0: 0.0, 1: 0.0}
            for row in curr_data:
                after_split_dict[row[key]][row['class']] += 1.0
            entropy_after = 0.0
            for val in self.field_values[key]:  # calculate entropy
                curr_tot = after_split_dict[val][0] + after_split_dict[val][1]
                entropy_curr = 0.0
                if curr_tot == 0.0:
                    continue
                for pk in after_split_dict[val].values():
                    if pk == 0.0:
                        continue
                    entropy_curr -= (pk / curr_tot) * self.log(pk / curr_tot, 2)
                entropy_after += entropy_curr * (curr_tot / float(len(curr_data)))

            if least_after_split_entropy == -1:
                least_after_split_entropy = entropy_after
                best_attr = key
            if entropy_after < least_after_split_entropy:  # get the attribute for which you get least new entropy after split
                least_after_split_entropy = entropy_after
                best_attr = key
        return best_attr

    def split_and_build(self, curr_data, node, depth, max_allowed_depth=8, min_len=8):
        node['attr'] = self.best_attr_to_split(curr_data)
        curr_classes = [c['class'] for c in curr_data]
        for val in self.field_values[node['attr']]:
            node[val] = {}
            new_data = filter(lambda x: x[node['attr']] == val, curr_data)  # get the nodes which have this attribute
            new_classes = [n['class'] for n in new_data]
            if not new_data:
                node[val]['output'] = int(curr_classes.count(1) > curr_classes.count(0))
            elif depth > max_allowed_depth or len(new_data) < min_len:  # prune if high depth or low length
                node[val]['output'] = int(new_classes.count(1) > new_classes.count(0))
            else:
                self.split_and_build(new_data, node[val], depth + 1)

    def train(self):  # train (call recursibe split function)
        self.split_and_build(self.processed_data['train'], self.tree, 0)

    def walk_and_predict(self, test_data, node):  # go recursively until you reach leaf and return the result in it.
        if 'output' in node:
            return node['output']
        else:
            return self.walk_and_predict(test_data, node[test_data[node['attr']]])

    def predict(self, test_file):
        self.extract_and_process(test_file, 'test')
        for row in self.processed_data['test']:
            self.predictions.append(self.walk_and_predict(row, self.tree))
        return self.predictions

    def get_accuracy(self):
        s, c = 0, 0
        for row in self.processed_data['test']:
            if row['class'] == self.predictions[c]:
                s += 1
            c += 1
        print 'Accuracy is', s / float(len(self.predictions))


dt = DecisionTree()
dt.extract_and_process('data/train.csv', 'train')
dt.train()
with open('ES15BTECH11018.model', 'w') as f:
    pickle.dump(dt, f)
print dt.predict('data/test.csv')
