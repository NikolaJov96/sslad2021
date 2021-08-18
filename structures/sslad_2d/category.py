from structures.sslad_2d.definitions import Colors


class Category:
    """
    Represents an annotation category
    """

    def __init__(self, name, category_id):
        """
        Initializes a category when loaded from the training annotations descriptor file
        """
        if category_id < 1 or category_id > 6:
            print('Invalid category id {}'.format(category_id))
            exit(1)

        self.name = name
        self.category_id = category_id

    def get_color(self):
        """
        Returns a color assigned to the category
        """
        return Colors.get_color_by_id(self.category_id)
