class Sample:
    def __init__(self, features, m):
        self.features = features
        self.weight = 1/m

    def set_weight(self, value):
        self.weight = value
