class Node():
    left = None
    right = None
    value = None
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    isArrayLeaf = False

    def __init__(self, value, split_axes, min_x, max_x, min_y, max_y):
        self.value = value
        self.split_axes = split_axes
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


    def getValue(self):
        return self.value
