from simpleai.search.local import hill_climbing, beam_best_first, hill_climbing_stochastic, beam, genetic
from clustering_problem import ClusteringProblem
from dlas import dla_hc


def ls_clustering_beam(X, beam_size=5):
    return beam(ClusteringProblem(X, True), beam_size).state.labels


def ls_clustering_genetic(X):
    return genetic(ClusteringProblem(X, True)).state.labels


def ls_clustering_dla(X, history_dept=5):
    return dla_hc(ClusteringProblem(X, True), history_dept)[0].labels


def ls_clustering_hc(X):
    return hill_climbing_stochastic(ClusteringProblem(X, True)).state.labels


if __name__ == '__main__':
    pass
    # y_pred = beam(ClusteringProblem(X, False), viewer=vw).state.labels
