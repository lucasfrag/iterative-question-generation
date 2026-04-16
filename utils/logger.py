import json
import os

class SimpleLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, data):
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")