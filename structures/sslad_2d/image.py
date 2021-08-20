import cv2
import os
from PIL import Image


class Image:
    """
    Represents a single image with its description, content and annotations
    """

    def __init__(self, image_path_prefix, image_data, dataset_type):
        """
        Initializes image description excluding annotations
        Image content is loaded when needed
        """

        self.file_name = os.path.join(image_path_prefix, image_data['file_name'])
        self.height = image_data['height']
        self.width = image_data['width']
        self.dataset_type = dataset_type
        self.annotations = []
        self.__cv2_img = None
        self.__pil_img = None

    def add_annotation(self, annotation):
        """
        Adds new annotation to the image
        """

        self.annotations.append(annotation)

    def get_cv2_img(self):
        """
        Loads the image content from file if not already loaded and returns it in OpenCV format
        """

        if self.__cv2_img is None:
            self.__cv2_img = cv2.imread(self.file_name)

        return self.__cv2_img

    def get_pil_img(self):
        """
        Loads the image content from file if not already loaded and returns it in PIL format
        """

        if self.__pil_img is None:
            self.__pil_img = Image.open(self.get_image_path(self.file_name)).convert("RGB")

        return self.__pil_img

    def draw_annotations(self, annotations=None, output_file=None):
        """
        Draws annotation bounding boxes over the image
        Custom annotations are drawn if provided
        Result is saved to the file if output file path is provided
        """

        if annotations is None:
            annotations = self.annotations

        # Make a copy of the original image
        bbox_img = self.get_cv2_img().copy()

        # Draw annotations
        for annotation in annotations:
            cv2.rectangle(
                bbox_img,
                (annotation.bbox[0], annotation.bbox[1]),
                (annotation.bbox[0] + annotation.bbox[2], annotation.bbox[1] + annotation.bbox[3]),
                annotation.category.get_color(),
                2
            )

        if output_file is not None:
            cv2.imwrite(output_file, bbox_img)

        return bbox_img

    @staticmethod
    def resize_to_width(img, new_width):
        """
        Returns new image resized to the new width, keeping the aspect ration
        """

        coefficient = new_width / img.shape[1]
        new_size = (int(img.shape[1] * coefficient), int(img.shape[0] * coefficient))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
