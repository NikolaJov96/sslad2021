import cv2

from structures.data_set import DataSet
from structures.image import Image


if __name__ == '__main__':
    """
    Cycles through training set images and displays them with drawn annotations 
    """

    data_set = DataSet()
    data_set.load()

    window_name = 'Annotated images'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for training_image_id in data_set.training_images:
        training_image: Image = data_set.training_images[training_image_id]
        img = training_image.draw_annotations()

        resize = 1000 / img.shape[1]
        new_size = (int(img.shape[1] * resize), int(img.shape[0] * resize))
        resized_img = cv2.resize(img, new_size)
        cv2.imshow(window_name, resized_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
