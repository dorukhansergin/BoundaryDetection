class WindowCropper:
    """
    A tool to crop positive and negative boundary images from a given shelf image
    """
    def __init__(self, margin):
        self.margin = margin

    def crop(self):
        raise NotImplementedError