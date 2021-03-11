import logging
import warnings

class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
