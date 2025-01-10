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
        for bin_list, value in bins_and_values:
            r, l = 0, len(bin_list) - 1
            while r <= l:
                mid = (r + l) // 2
                if value <= bin_list[mid]:
                    index = mid
                    l = mid - 1
                else:
                    r = mid + 1
            self.digitized_state.append(index)