import cv2
import os
from PIL import Image as PILImage


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
        self.image_id = image_data['id']
        self.height = image_data['height']
        self.width = image_data['width']
        self.dataset_type = dataset_type
        self.is_annotated = False
        self.annotations = []

    def add_annotation(self, annotation):
        """
        Adds new annotation to the image
        """

        self.annotations.append(annotation)
        self.is_annotated = True

    def add_annotations(self, annotations):
        """
        Concatenates list of annotations to the image annotations
        """

        if not isinstance(annotations, list):
            print('Image.add_annotations expects a list, received {}'.format(type(annotations)))
            exit(1)

        self.annotations += annotations
        self.is_annotated = True

    def get_cv2_img(self):
        """
        Loads the image content from file if not already loaded and returns it in OpenCV format
        """

        return cv2.imread(self.file_name)

    def get_pil_img(self):
        """
        Loads the image content from file if not already loaded and returns it in PIL format
        """

        return PILImage.open(self.file_name).convert("RGB")

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
                (round(annotation.x), round(annotation.y)),
                (round(annotation.x + annotation.w), round(annotation.y + annotation.h)),
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
