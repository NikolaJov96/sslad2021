class Annotation:
    """
    Represents a single bounding box annotation
    """

    def __init__(self, category, annotation_data):
        """
        Initializes annotation description
        """
        self.category = category
        self.bbox = annotation_data['bbox']
        self.area = annotation_data['area']
        self.annotation_id = annotation_data['id']
