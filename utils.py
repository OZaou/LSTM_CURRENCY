# utils.py

import os
import logging
import yaml

def setup_logging(log_file, level=logging.INFO):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
