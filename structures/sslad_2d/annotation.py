class Annotation:
    """
    Represents a single bounding box annotation
    """

    def __init__(self, category, annotation_data):
        """
        Initializes annotation description
        """

        self.category = category
        self.x = annotation_data['bbox'][0]
        self.y = annotation_data['bbox'][1]
        self.w = annotation_data['bbox'][2]
        self.h = annotation_data['bbox'][3]
        self.area = annotation_data['area']
