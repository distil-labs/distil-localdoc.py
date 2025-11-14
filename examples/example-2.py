class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def process(self, raw_data):
        cleaned = [x for x in raw_data if x is not None]
        return [self.transform(x) for x in cleaned]