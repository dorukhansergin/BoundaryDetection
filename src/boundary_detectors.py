class BoundaryDetector:
    """
    Generic boundary detection class. All different problem solution methods will be considered under this class
    """
    def predict_boundaries(self):
        """
        The end goal function that takes an image as input and outputs the pixel locations of predicted boundaries
        """
        raise NotImplementedError


class SlidingWindowBoundaryDetector(BoundaryDetector):
    def __init__(self, window_predictor):
        self.window_predictor = window_predictor

    def predict_boundaries(self):
        # TODO:
        raise NotImplementedError

