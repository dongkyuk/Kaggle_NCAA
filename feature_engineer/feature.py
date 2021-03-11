import pandas as pd
from abc import ABC, abstractmethod
from utils.time_function import time_function

class Feature(ABC):        
    @abstractmethod
    def make_features(self):
        pass

    def load_features(self):
        self.feature = pd.read_csv(self.feature_save_path)

    @abstractmethod    
    def merge_features(self):
        pass

    @time_function
    def run(self):
        if self.load:
            self.load_features()
        else:
            self.make_features()
        return self.merge_features()

