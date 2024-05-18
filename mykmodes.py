import sys
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger
from numpy.random import RandomState
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array


def matching_dissimilarity(a: Any, b: Any, **_):
    return np.sum(a != b, axis=1)


def pandas_to_numpy(x) -> np.ndarray:
    if "pandas" in str(x.__class__):
        return x.values
    return x


def get_max_value_key(dictionary: dict) -> np.ndarray:
    values, keys = np.array(list(dictionary.values())), np.array(
        list(dictionary.keys())
    )
    max_value = np.where(values == np.max(values))[0]
    if len(max_value) == 1:
        return keys[max_value[0]]
    return keys[max_value[np.argmin(keys[max_value])]]


def encode_features(
    X: np.ndarray, encoding_mapping: list | None = None
) -> tuple[np.ndarray, list]:
    if encoding_mapping is None:
        count_encoding_mapping = True
        encoding_mapping = []
    else:
        count_encoding_mapping = False

    X_encoded = np.zeros(X.shape, dtype="int32")
    for i in range(X.shape[1]):
        if count_encoding_mapping:
            encoded_column = {val: j for j, val in enumerate(np.unique(X[:, i]))}
            encoding_mapping.append(encoded_column)
        X_encoded[:, i] = np.array([encoding_mapping[i].get(x, -1) for x in X[:, i]])

    return X_encoded, encoding_mapping


def decode_centroids(encoded: np.ndarray, mapping: list) -> np.ndarray:
    decoded_centroids = []
    for i in range(encoded.shape[1]):
        inversed_mapping = {v: k for k, v in mapping[i].items()}
        decoded_centroids.append(
            np.vectorize(inversed_mapping.__getitem__)(encoded[:, i])
        )
    return np.atleast_2d(np.array(decoded_centroids)).T


def get_unique_rows(a: np.ndarray) -> np.ndarray:
    return np.vstack(list({tuple(row) for row in a}))


class MyKModes:
    def __init__(
        self,
        clusters_amount: int,
        max_iterations: int = 100,
        runs_amount: int = 10,
        logging: bool = True,
        random_state: int | RandomState | None = None,
    ):
        logger_handlers = [
            {
                "sink": sys.stdout,
                "level": "DEBUG" if logging else "INFO",
                "format": "<level>level={level} {message}</level>",
            }
        ]
        logger.configure(handlers=logger_handlers)
        self.clusters_amount = clusters_amount
        self.max_iterations = max_iterations
        self.runs_amount = runs_amount
        self.logging = logging
        self.random_state = random_state
        self._fitted = False

    def fit(self, X):
        X = pandas_to_numpy(X)

        (
            self._enc_cluster_centroids,
            self._enc_map,
            self.labels_,
            self.cost_,
            self.n_iter_,
            self.epoch_costs_,
        ) = self.__k_modes(X, check_random_state(self.random_state))
        self._fitted = True
        return self

    def fit_predict(self, X) -> np.ndarray:
        return self.fit(X).predict(X)

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise Exception("Model is not fitted")

        X = pandas_to_numpy(X)
        X = check_array(X, dtype=None)
        X, _ = encode_features(X, encoding_mapping=self._enc_map)
        return self.__labels_cost(X, self._enc_cluster_centroids)[0]

    @property
    def cluster_centroids_(self) -> np.ndarray:
        if not self._fitted:
            raise Exception("Model is not fitted")
        return decode_centroids(self._enc_cluster_centroids, self._enc_map)

    def __k_modes(
        self,
        X: np.ndarray,
        random_state: RandomState,
    ) -> tuple[np.ndarray, list, np.ndarray | None, float, int, list[float]]:

        X = check_array(X, dtype=None)

        X, encoding_mapping = encode_features(X)

        points_amount, attributes_amount = X.shape

        unique_rows = get_unique_rows(X)
        unique_amount = unique_rows.shape[0]
        if unique_amount <= self.clusters_amount:
            self.max_iterations = 0
            self.runs_amount = 1
            self.clusters_amount = unique_amount

        results = []
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.runs_amount)
        for run_number in range(self.runs_amount):
            results.append(
                self.__k_modes_single_run(
                    X,
                    points_amount,
                    attributes_amount,
                    run_number,
                    seeds[run_number],
                )
            )
        all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(
            *results
        )

        best_run = np.argmin(all_costs)
        if self.runs_amount > 1:
            logger.debug(f"Best run was number {best_run + 1}")

        return (
            all_centroids[best_run],
            encoding_mapping,
            all_labels[best_run],
            all_costs[best_run],
            all_n_iters[best_run],
            all_epoch_costs[best_run],
        )

    def __k_modes_single_run(
        self,
        X: np.ndarray,
        points_amount: int,
        attributes_amount: int,
        run_number: int,
        random_state: RandomState,
    ) -> tuple[np.ndarray, np.ndarray | None, float, int, list[float]]:
        random_state = check_random_state(random_state)
        logger.debug("Init: initializing centroids")

        seeds = random_state.choice(range(points_amount), self.clusters_amount)
        centroids = X[seeds]

        logger.debug("Initializing clusters...")

        membership = np.zeros((self.clusters_amount, points_amount), dtype=np.bool_)

        cluster_attribute_frequences = [
            [defaultdict(int) for _ in range(attributes_amount)]
            for _ in range(self.clusters_amount)
        ]
        for i, point in enumerate(X):
            weight = 1
            clust = np.argmin(
                matching_dissimilarity(centroids, point, X=X, membship=membership)
            )
            membership[clust, i] = 1
            for j, attribute in enumerate(point):
                cluster_attribute_frequences[clust][j][attribute] += weight

        for i in range(self.clusters_amount):
            for j in range(attributes_amount):
                if sum(membership[i]) == 0:
                    centroids[i, j] = random_state.choice(X[:, j])
                else:
                    centroids[i, j] = get_max_value_key(
                        cluster_attribute_frequences[i][j]
                    )

        logger.debug("Starting clustering iterations...")

        iteration = 0
        labels = None
        is_converged = False

        _, cost = self.__labels_cost(X, centroids, membership)

        epoch_costs = [cost]

        while iteration < self.max_iterations and not is_converged:
            iteration += 1
            centroids, cluster_attribute_frequences, membership, moves = (
                self.__k_modes_iteration(
                    X,
                    centroids,
                    cluster_attribute_frequences,
                    membership,
                    random_state,
                )
            )
            labels, ncost = self.__labels_cost(X, centroids, membership)
            is_converged = (moves == 0) or (ncost >= cost)
            epoch_costs.append(ncost)
            cost = ncost
            logger.debug(
                f"Run {run_number + 1} iteration: {iteration}/{self.max_iterations}, "
                f"moves: {moves}, cost: {cost}"
            )

        return centroids, labels, cost, iteration, epoch_costs

    def __k_modes_iteration(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
        cluster_attribute_frequencies: list[list],
        membership: np.ndarray,
        random_state: RandomState,
    ) -> tuple[np.ndarray, list[list], np.ndarray, int]:
        moves = 0
        for i, point in enumerate(X):
            weight = 1
            cluster = np.argmin(
                matching_dissimilarity(centroids, point, X=X, membship=membership)
            )
            if membership[cluster, i]:
                continue

            moves += 1
            old_cluster = np.argwhere(membership[:, i])[0][0]

            cluster_attribute_frequencies, membership, centroids = (
                self._move_point_categorical(
                    point,
                    i,
                    cluster,
                    old_cluster,
                    cluster_attribute_frequencies,
                    membership,
                    centroids,
                    weight,
                )
            )
            if not membership[old_cluster, :].any():
                from_clust = membership.sum(axis=1).argmax()
                choices = [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
                random_i = random_state.choice(choices)

                cluster_attribute_frequencies, membership, centroids = (
                    self._move_point_categorical(
                        X[random_i],
                        random_i,
                        old_cluster,
                        from_clust,
                        cluster_attribute_frequencies,
                        membership,
                        centroids,
                        weight,
                    )
                )

        return centroids, cluster_attribute_frequencies, membership, moves

    def __labels_cost(
        self, X: np.ndarray, centroids: np.ndarray, membship: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        X = check_array(X)

        points_amount = X.shape[0]
        cost = 0.0
        labels = np.empty(points_amount, dtype=np.uint16)
        for ipoint, curpoint in enumerate(X):
            diss = matching_dissimilarity(centroids, curpoint, X=X, membship=membship)
            clust = np.argmin(diss)
            labels[ipoint] = clust
            cost += diss[clust]
        return labels, cost

    def _move_point_categorical(
        self,
        point: np.ndarray,
        random_i: int,
        to_cluster: np.intp,
        from_cluster: np.intp,
        cluster_attribute_frequency: list[list],
        membership: np.ndarray,
        centroids: np.ndarray,
        weight: int,
    ) -> tuple[list[list], np.ndarray, np.ndarray]:
        membership[to_cluster, random_i] = 1
        membership[from_cluster, random_i] = 0
        for i, attribute in enumerate(point):
            to_attribute_counts = cluster_attribute_frequency[to_cluster][i]
            from_attribute_counts = cluster_attribute_frequency[from_cluster][i]

            to_attribute_counts[attribute] += weight

            current_attribute_value_frequency = to_attribute_counts[attribute]
            current_centroid_value = centroids[to_cluster][i]
            current_centroid_frequency = to_attribute_counts[current_centroid_value]
            if current_centroid_frequency < current_attribute_value_frequency:
                centroids[to_cluster][i] = attribute

            from_attribute_counts[attribute] -= weight

            old_centroid_value = centroids[from_cluster][i]
            if old_centroid_value == attribute:
                centroids[from_cluster][i] = get_max_value_key(from_attribute_counts)

        return cluster_attribute_frequency, membership, centroids
