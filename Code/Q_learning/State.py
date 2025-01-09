class State:
    def __init__(self, storage_level, price, hour, day):
        self.storage_level = storage_level
        self.price = price
        self.hour = hour
        self.day = day

        self.digitized_state = []

    def digitize(self, bins):
        bins_and_values = [
            (bins[0], self.storage_level),
            (bins[1], self.price),
            (bins[2], self.hour),
            (bins[3], self.day),
        ]

        self.digitized_state = []
        for bins, value in bins_and_values:
            self.digitized_state.append(next(i for i, b in enumerate(bins) if value <= b))