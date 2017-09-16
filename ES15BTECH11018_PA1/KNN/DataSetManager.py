import csv
import random
import os.path
from datetime import datetime


class DataSetManager(object):

    def __init__(self, file_dict, fields, data_range, class_range, ndata_points): # initialize required data
        self.file_dict = file_dict
        self.fields = fields
        self.data_range = data_range
        self.class_range = class_range
        self.ndata_points = ndata_points
        self.data = []
        random.seed(datetime.now())

    def generate(self): # generate if files do not exist
        if os.path.isfile(self.file_dict['train']):
            print "File Already exists, no need to generate again!"
            return

        for d in xrange(1, self.ndata_points):
            row = {}
            for field in self.fields[1:-1]:
                row[field] = random.randint( # Random integer in range
                    self.data_range[0], self.data_range[1])
            row['class'] = random.randint(
                self.class_range[0], self.class_range[1])
            self.data.append(row) # append the row in final list

    def get_dicts(self, reader): # get dictionary from read row
        ref = []
        reader.next()
        for row in reader:
            for key in row:
                if key is not 'id':
                    row[key] = int(row[key]) #convert to int
            ref.append(row)
        return ref

    def load(self): # load from the csv files generated
        with open(self.file_dict['train'], 'r') as train_file, open(self.file_dict['test'], 'r') as test_file:
            train_reader = csv.DictReader(train_file, fieldnames=self.fields)
            test_reader = csv.DictReader(test_file, fieldnames=self.fields)
            self.training_set = self.get_dicts(train_reader)
            self.test_set = self.get_dicts(test_reader)

    def splitAndWrite(self, percent): # split according to percentage and update data
        if os.path.isfile(self.file_dict['train']):
            return
        with open(self.file_dict['train'], 'w') as train_file, open(self.file_dict['test'], 'w') as test_file:
            train_writer = csv.DictWriter(train_file, fieldnames=self.fields)
            test_writer = csv.DictWriter(test_file, fieldnames=self.fields)

            random.shuffle(self.data) #shuffle randomly
            for d in range(len(self.data)):
                self.data[d]['id'] = (d + 1)

            train_writer.writeheader()
            train_writer.writerows(self.data[: int(percent * self.ndata_points)]) # slice according to percent *length

            test_writer.writeheader()
            test_writer.writerows(self.data[int(percent * self.ndata_points):])
