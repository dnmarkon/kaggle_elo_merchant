import json
import pandas as pd
from src.data.ABCDataProvider import ABCDataProvider


class FileDataProvider(ABCDataProvider):

    def __init__(self, train_file, parameters_file):
        self.train_file = train_file
        self.parameters_file = parameters_file

    def load_train(self):
        return pd.read_csv(self.train_file)

    def save_parameters(self, parameters):
        with open(self.parameters_file, 'w') as outfile:
            json.dump(parameters, outfile)


