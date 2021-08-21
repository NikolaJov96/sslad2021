class Annotation:
    """
    Represents a single bounding box annotation
    """

    def __init__(self, category, x, y, w, h):
        """
        Initializes annotation description
        """

        self.category = category
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h

    def get_array(self):
        """
        Return bbox as 4-element array
        """

        return [self.x, self.y, self.w, self.h]

    @staticmethod
    def from_annotation_data(category, annotation_data):
        """
        Creates an annotation from annotation data dict loaded from dataset file
        """

        area = annotation_data['area']
        w = annotation_data['bbox'][2]
        h = annotation_data['bbox'][3]

        if area - w * h != 0:
            print('area {} not matching with {} and height {}'.format(area, w, h))
            exit(1)

        return Annotation(
            category,
            annotation_data['bbox'][0],
            annotation_data['bbox'][1],
            annotation_data['bbox'][2],
            annotation_data['bbox'][3]
        )
