import sys

from competition_models.trainer import Trainer


def main():
    """
    Run the trainer prediction on the validation or testing datasets
    and save results to competition submission compatible json format
    """

    if not len(sys.argv) == 3 or sys.argv[2] not in ['validation', 'testing']:
        print('usage: python3 generate_predictions.py session_id "validation"|"testing"')
        exit(1)

    training_session = sys.argv[1]
    trainer = Trainer(training_session)
    trainer.generate_output_predictions(sys.argv[2])


if __name__ == '__main__':
    main()
