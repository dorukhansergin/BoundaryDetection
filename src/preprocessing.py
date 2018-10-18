class WindowCropper:
    """
    A tool to crop positive and negative boundary images from a given shelf image
    """
    def __init__(self, margin):
        # make sure the margin is odd
        assert margin % 2 == 1, "margin ({}) is not odd, please choose an odd margin ".format(margin)
        self.margin = margin

    def crop_all(self, image, boundary_locations):
        return self.crop_positives(image, boundary_locations), self.crop_negatives(image, boundary_locations)

    def crop_single(self, image, location):
        """
        :param image: numpy array, the whole shelf image to be cropped
        :param location: integer, the index of the center of crop
        :return:
        An image of width margin in which location is centered
        """
        arm_length = (self.margin - 1) / 2
        return image[:,location-arm_length:location+arm_length]

    def crop_positives(self, image, boundary_locations):
        return list(map(lambda location: self.crop_single(image, location), boundary_locations))

    def crop_negatives(self, image, boundary_locations):
        return [self.crop_single(image, int((boundary_locations[i] + boundary_locations[i+1])/2))
         for i in range(len(boundary_locations) - 1)]

