from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import pairwise_distances
from hdbscan import validity_index
import copy
import random
import numpy as np
from simpleai.search import SearchProblem


def combine_metric_values(values):
    len_values = len(values)
    return 1 / sum(map(lambda x: ((1 / len_values) * (1 / x)), values))


class ClusteringProblem(SearchProblem):
    class State:
        def __init__(self, labels: list[int], label_counts: dict = None):
            self.labels = labels
            self.label_counts = label_counts if label_counts else {label: labels.count(label) for label in set(labels)}

        def __str__(self):
            return str(self.labels)

    def __init__(self, samples: np.ndarray, fixed_cluster_count: bool = False):
        self.samples = samples
        self.fixed_cluster_count = fixed_cluster_count
        self.n_samples = samples.shape[0]
        self.dist_matrix = pairwise_distances(samples)
        initial_state = ClusteringProblem.State(
            labels=[0] * self.n_samples,
            label_counts={0: self.n_samples},
        )
        super(ClusteringProblem, self).__init__(initial_state=initial_state)

    def __assignable_labels(self, state: State) -> list[int]:
        labels = []
        zero_count_label_added = False
        for label, count in state.label_counts.items():

            if count > 0:
                labels.append(label)

            if count == 0 and not zero_count_label_added:
                labels.append(label)
                zero_count_label_added = True

        if not zero_count_label_added and not self.fixed_cluster_count:
            labels.append(max(labels) + 1)

        return labels

    def __nr_of_labels(self, state: State) -> int:
        return len(list(filter(lambda c: c > 0, state.label_counts.values())))

    def actions(self, state: State):
        labels = self.__assignable_labels(state)
        # TODO: It is not allowed to remove the last sample from a cluster when fixed_cluster_count is set!

        # (sample_i, label) =: set sample_i to label
        actions = [(sample_i, label) for sample_i in range(self.n_samples) for label in labels if
                   state.labels[sample_i] != label]
        return actions

    def random_action(self, state: State):
        labels = self.__assignable_labels(state)

        sample_i = random.randint(0, self.n_samples - 1)
        label = random.choice(labels)

        while state.labels[sample_i] == label:
            sample_i = random.randint(0, self.n_samples - 1)
            label = random.choice(labels)

        return sample_i, label

    def mutate(self, state) -> State:
        return self.result(state, self.random_action(state))

    def crossover(self, state1, state2) -> State:
        new_labels = [0] * self.n_samples

        for i in range(self.n_samples):
            new_labels[i] = state1.labels[i] if random.random() < 0.5 else state2.labels[i]

        return ClusteringProblem.State(labels=new_labels)

    def result(self, state: State, action):
        sample_i, label = action

        state = copy.deepcopy(state)
        state.label_counts[state.labels[sample_i]] -= 1
        state.labels[sample_i] = label

        if label not in state.label_counts:
            state.label_counts[label] = 0

        state.label_counts[label] += 1
        return state

    def value(self, state: State):
        mean_silhouette_coefficient = silhouette_score(self.samples, state.labels) if self.__nr_of_labels(
            state) >= 2 else 0
        # chs = calinski_harabasz_score(self.samples, state.labels)
        dbcv = validity_index(self.samples, np.array(state.labels), )
        return combine_metric_values([mean_silhouette_coefficient, dbcv])

    def generate_random_state(self):
        return ClusteringProblem.State(labels=random.choices([0, 1], k=self.n_samples))
