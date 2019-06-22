import json
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(DIR_PATH, "mapillary.json"), "r") as f:
    data = json.load(f)

labels = data["labels"]
color = [tuple(label['color']) for label in labels]
n_labels = len(labels)
labels2color = dict(zip(range(n_labels), color))
