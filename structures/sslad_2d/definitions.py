import enum


class DataSetTypes(enum.Enum):
    """
    Enum containing existing data set types
    """
    TRAINING = enum.auto()
    VALIDATION = enum.auto()
    TESTING = enum.auto()


class Colors(enum.Enum):
    """
    Enum representing bounding box drawing colors
    """

    RED = enum.auto()
    BLUE = enum.auto()
    GREEN = enum.auto()
    YELLOW = enum.auto()
    CYAN = enum.auto()
    FUCHSIA = enum.auto()

    COLOR_LIST = ['red', 'blue', 'green', 'yellow', 'cyan', 'fuchsia']

    @staticmethod
    def get_color_by_id(color_id):
        """
        Returns the BGR color of the provided enum value
        """

        if color_id == Colors.RED.value:
            return 0, 0, 255
        elif color_id == Colors.BLUE.value:
            return 255, 0, 0
        elif color_id == Colors.GREEN.value:
            return 0, 255, 0
        elif color_id == Colors.YELLOW.value:
            return 0, 255, 255
        elif color_id == Colors.CYAN.value:
            return 255, 255, 0
        elif color_id == Colors.FUCHSIA.value:
            return 255, 0, 255
        else:
            print('Invalid color id {}'.format(color_id))
            exit(1)

    @staticmethod
    def get_color_by_name(color_name):
        """
        Allow getting a color by name
        """

        try:
            color_id = Colors.COLOR_LIST.index(color_name)
        except ValueError:
            print('Invalid color name {}'.format(color_name))
            exit(1)

        return Colors.get_color_by_id(color_id)
