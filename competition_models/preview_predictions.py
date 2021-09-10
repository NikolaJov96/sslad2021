import cv2
import sys

from competition_models.trainer import Trainer
from structures.sslad_2d.definitions import SSLADDatasetTypes
from structures.sslad_2d.image import Image
from structures.sslad_2d.sslad_dataset import SSLADDataset


def main():
    """
    Load and preview prediced validation or testing annotations,
    ready to be submitted for the competition
    """

    if not len(sys.argv) == 3 or sys.argv[2] not in ['validation', 'testing']:
        print('usage: python3 preview_predictions.py session_id "validation"|"testing"')
        exit(1)

    is_validation = sys.argv[2] == 'validation'

    training_session = sys.argv[1]
    trainer = Trainer(training_session)
    data_file = trainer.output_prediction_path(sys.argv[2])

    dataset = SSLADDataset()
    images = []
    if is_validation:
        dataset.load(filter_no_annotations=False, validation_data_file=data_file)
        images = dataset.get_subset(SSLADDatasetTypes.VALIDATION)
    else:
        dataset.load(filter_no_annotations=False, test_data_file=data_file)
        images = dataset.get_subset(SSLADDatasetTypes.TESTING)

    window_name = 'Annotated images {}'.format(sys.argv[2])
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i, image in enumerate(images):

        print('\rimage {}/{}'.format(i, len(images)), end='')

        img = image.draw_annotations()

        resized_img = Image.resize_to_width(img, 1000)

        cv2.imshow(window_name, resized_img)
        # Exit on esc
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
