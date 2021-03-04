from abc import ABC, abstractmethod

class Feature(ABC):
    @abstractmethod
    def make_features(self):
        pass

    @abstractmethod    
    def merge_features(self):
        pass
    
    def run(self):
        self.make_features()
        return self.merge_features()

