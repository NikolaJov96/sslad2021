import matplotlib.pyplot as plt
import sys

from competition_models.trainer import Trainer
from structures.sslad_2d.sslad_dataset import SSLADDataset


def main():
    """
    Creates a graph of a category AP for each training iteration for all categories
    """

    if len(sys.argv) != 2:
        print('usage: python3 session_analysis.py session_id')
    training_session = sys.argv[1]

    dataset = SSLADDataset()
    dataset.load()

    trainer = Trainer(training_session)

    average_precisions = [[] for _ in range(7)]

    print('iteration 0')
    print('result {}'.format(trainer.initial_log["evaluation"]))
    for ap_id in range(7):
        average_precisions[ap_id].append(trainer.initial_log["evaluation"][ap_id])

    for i, log in enumerate(trainer.iteration_logs):
        iteration = i + 1
        print('iteration {}'.format(iteration))
        for j, unlabeled_tested in enumerate(log["unlabeled_tested"]):
            print('model {} result {}'.format(j, unlabeled_tested['evaluation']))

        best_model_id = trainer.get_best_sub_model_id(iteration)
        print('best model id {}'.format(best_model_id))

        result = log["unlabeled_tested"][best_model_id]["evaluation"]

        for ap_id in range(7):
            average_precisions[ap_id].append(result[ap_id])

    # Plot the graph
    iteration_numbers = list(range(len(trainer.iteration_logs) + 1))
    labels = ['Mean ap'] + ['{} ap'.format(dataset.categories[category_id].name) for category_id in dataset.categories]

    fig, ax = plt.subplots()

    for ap_id in range(7):
        ax.plot(iteration_numbers, average_precisions[ap_id], label=labels[ap_id])

    ax.set(
        xlabel='iteration',
        ylabel='average precision',
        title='Average precisions'
    )
    ax.xaxis.set_ticks(iteration_numbers)
    ax.set_ylim(min(0.0, min(average_precisions[0])) - 0.05, 1.05)
    ax.grid()
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
