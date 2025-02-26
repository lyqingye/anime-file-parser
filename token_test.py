import json
from transformers import AutoTokenizer
import pandas as pd

from train import convert_to_ner_format, load_data


data = load_data('data/anime_ner_training_data.json')
convert_to_ner_format(data, print_data=True)