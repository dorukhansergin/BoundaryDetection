class Plotter:
    def __init__(self):
        self.axis = None

    def set_axis(self, axis):
        self.axis = axis

    def add_half_lines_above(self, locations):
        for loc in locations:
            self.axis.axvline(x=loc, ymin=0.5, color='r')

    def add_half_lines_below(self, locations):
        for loc in locations:
            self.axis.axvline(x=loc, ymax=0.5, color='r')

    def add_full_lines(self, locations, mode='ground_truth'):
        if mode == 'ground_truth':
            color = 'y'
        elif mode == 'true_positive':
            color = 'g'
        for loc in locations:
            self.axis.axvline(x=loc, color=color)