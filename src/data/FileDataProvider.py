import csv
import json

import pandas as pd
from src.data.ABCDataProvider import ABCDataProvider
import pickle


class FileDataProvider(ABCDataProvider):

    def __init__(self, train_file, parameters_file, model_file, submission_file):
        self.train_file = train_file
        self.parameters_file = parameters_file
        self.model_file = model_file
        self.submission_file = submission_file

    def load_train(self):
        return pd.read_csv(self.train_file)

    def save_best_parameters(self, parameters):
        with open(self.parameters_file, 'w') as outfile:
            json.dump(parameters, outfile)

    def save_model(self, model):
        with open(self.model_file, 'wb') as outfile:
            pickle.dump(obj=model, file=outfile)

    def load_model(self):
        with open(self.model_file, 'rb') as m_file:
            return pickle.load(m_file)

    def save_submission(self, x_test, predictions):
        with open(self.submission_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['card_id', 'target'])
            for i, row in x_test.iterrows():
                writer.writerow([row['card_id'], predictions[i]])
