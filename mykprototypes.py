from collections import defaultdict

import numpy as np
from numpy.random import RandomState
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from mykmodes import (
    MyKModes,
    decode_centroids,
    encode_features,
    get_max_value_key,
    get_unique_rows,
    matching_dissimilarity,
    pandas_to_numpy,
)

MAX_INIT_TRIES = 20
RAISE_INIT_TRIES = 100


def euclidean_dissimilarity(a, b, **_):
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)


class MyKPrototypes(MyKModes):
    def __init__(
        self,
        clusters_amount: int,
        max_iterations: int = 100,
        runs_amount: int = 10,
        gamma=None,
        logging: bool = False,
        random_state: int | RandomState | None = None,
    ):

        super(MyKPrototypes, self).__init__(
            clusters_amount,
            max_iterations,
            runs_amount=runs_amount,
            logging=logging,
            random_state=random_state,
        )
        self.gamma = gamma

    def fit(
        self,
        X,
        categorical: int | list[int] | tuple[int] | None = None,
        sample_weight=None,
    ):
        if categorical is not None:
            if not isinstance(categorical, (int, list, tuple)):
                raise Exception(
                    "Categories must be passed as int, tuple[int] or list[int]"
                )

        X = pandas_to_numpy(X)
        _validate_sample_weight(
            sample_weight, n_samples=X.shape[0], n_clusters=self.clusters_amount
        )
        random_state = check_random_state(self.random_state)
        self.sample_weight = sample_weight
        (
            self._enc_cluster_centroids,
            self._enc_map,
            self.labels_,
            self.cost_,
            self.n_iter_,
            self.epoch_costs_,
            self.gamma,
        ) = self.__k_prototypes(
            X,
            categorical,
            self.gamma,
            random_state,
        )
        self._fitted = True
        return self

    def predict(
        self, X, categorical: int | list[int] | tuple[int] | None = None, **kwargs
    ):
        if not self._fitted:
            raise Exception("Model is not fitted")

        if categorical is not None:
            if not isinstance(categorical, (int, list, tuple)):
                raise Exception(
                    "Categories must be passed as int, tuple[int] or list[int]"
                )

        X = pandas_to_numpy(X)
        X_numerical, X_categorical = self._split_num_cat(X, categorical)
        X_numerical, X_categorical = check_array(X_numerical), check_array(
            X_categorical, dtype=None
        )
        X_categorical, _ = encode_features(
            X_categorical, encoding_mapping=self._enc_map
        )
        return self.__labels_cost(
            X_numerical,
            X_categorical,
            self._enc_cluster_centroids,
            self.gamma,
        )[0]

    def fit_predict(
        self,
        X,
        categorical: int | list[int] | tuple[int] | None = None,
        sample_weight=None,
    ) -> np.ndarray:
        return self.fit(
            X, categorical=categorical, sample_weight=sample_weight
        ).predict(X, categorical=categorical)

    @property
    def cluster_centroids_(self) -> np.ndarray:
        if not self._fitted:
            raise Exception("Model is not fitted")
        return np.hstack(
            (
                self._enc_cluster_centroids[0],
                decode_centroids(self._enc_cluster_centroids[1], self._enc_map),
            )
        )

    def __k_prototypes(
        self,
        X: np.ndarray,
        categorical: int | list[int] | tuple[int] | None,
        gamma: float | None,
        random_state: RandomState,
    ):
        random_state = check_random_state(random_state)
        if categorical is None or not categorical:
            raise Exception("No categorical data specified - does not supported")
        if isinstance(categorical, int):
            categorical = [categorical]

        if max(categorical) > X.shape[1]:
            raise Exception("Categorical column index out of bounds")

        categorical_attributes_amount = len(categorical)
        numerical_attributes_amount = X.shape[1] - categorical_attributes_amount
        points_amount = X.shape[0]

        X_numerical, X_categorical = self._split_num_cat(X, categorical)
        X_numerical, X_categorical = check_array(X_numerical), check_array(
            X_categorical, dtype=None
        )

        X_categorical, enc_map = encode_features(X_categorical)

        unique = get_unique_rows(X)
        unique_amount = unique.shape[0]

        if unique_amount <= self.clusters_amount:
            self.max_iterations = 0
            self.runs_amount = 1
            self.clusters_amount = unique_amount
            init = list(self._split_num_cat(unique, categorical))
            init[1], _ = encode_features(init[1], enc_map)

        if gamma is None:
            gamma = 0.5 * np.mean(X_numerical.std(axis=0))

        results = []
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.runs_amount)
        for run_number in range(self.runs_amount):
            results.append(
                self.__k_prototypes_single_run(
                    X_numerical,
                    X_categorical,
                    numerical_attributes_amount,
                    categorical_attributes_amount,
                    points_amount,
                    gamma,  # type: ignore
                    run_number,
                    seeds[run_number],
                )
            )
        all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(
            *results
        )

        best_run = np.argmin(all_costs)
        if self.runs_amount > 1 and self.logging:
            print(f"Best run was number {best_run + 1}")

        return (
            all_centroids[best_run],
            enc_map,
            all_labels[best_run],
            all_costs[best_run],
            all_n_iters[best_run],
            all_epoch_costs[best_run],
            gamma,
        )

    def __k_prototypes_single_run(
        self,
        X_numerical: np.ndarray,
        X_categorical: np.ndarray,
        numerical_attributes_amount: int,
        categorical_attributes_amount: int,
        points_amount: int,
        gamma: float,
        run_number: int,
        random_state: RandomState,
    ):
        random_state = check_random_state(random_state)
        init_tries = 0
        while True:
            init_tries += 1
            if self.logging:
                print("Init: initializing centroids")
            seeds = random_state.choice(range(points_amount), self.clusters_amount)
            centroids = X_categorical[seeds]
            mean_x_numerical = np.mean(X_numerical, axis=0)
            std_x_numerical = np.std(X_numerical, axis=0)
            centroids = [
                mean_x_numerical
                + random_state.randn(self.clusters_amount, numerical_attributes_amount)
                * std_x_numerical,
                centroids,
            ]
            if self.logging:
                print("Initializing clusters...")
            membership = np.zeros((self.clusters_amount, points_amount), dtype=np.bool_)
            attributes_sums = np.zeros(
                (self.clusters_amount, numerical_attributes_amount), dtype=np.float64
            )
            memberships_sums = np.zeros(self.clusters_amount, dtype=np.float64)
            attributes_frequencies = [
                [defaultdict(float) for _ in range(categorical_attributes_amount)]
                for _ in range(self.clusters_amount)
            ]
            for point in range(points_amount):
                weight = (
                    self.sample_weight[point] if self.sample_weight is not None else 1
                )
                cluster = np.argmin(
                    euclidean_dissimilarity(centroids[0], X_numerical[point])
                    + gamma
                    * matching_dissimilarity(
                        centroids[1],
                        X_categorical[point],
                        X=X_categorical,
                        membship=membership,
                    )
                )
                membership[cluster, point] = 1
                memberships_sums[cluster] += weight
                for j, attribute in enumerate(X_numerical[point]):
                    attributes_sums[cluster, j] += attribute * weight
                for j, attribute in enumerate(X_categorical[point]):
                    attributes_frequencies[cluster][j][attribute] += weight

            if membership.sum(axis=1).min() > 0:
                break

            elif init_tries == RAISE_INIT_TRIES:
                raise Exception("Reached init tries limit")
        for i in range(self.clusters_amount):
            for j in range(numerical_attributes_amount):
                centroids[0][i, j] = attributes_sums[i, j] / memberships_sums[i]
            for j in range(categorical_attributes_amount):
                centroids[1][i, j] = get_max_value_key(attributes_frequencies[i][j])

        if self.logging:
            print("Starting clustering iterations...")

        iteration = 0
        labels = None
        is_converged = False

        _, cost = self.__labels_cost(
            X_numerical,
            X_categorical,
            centroids,
            gamma,
            membership,
        )

        epoch_costs = [cost]

        while iteration < self.max_iterations and not is_converged:
            iteration += 1
            (
                centroids,
                attributes_sums,
                memberships_sums,
                attributes_frequencies,
                membership,
                moves,
            ) = self.__k_prototypes_iteration(
                X_numerical,
                X_categorical,
                centroids,
                attributes_sums,
                memberships_sums,
                attributes_frequencies,
                membership,
                gamma,
                random_state,
            )

            labels, ncost = self.__labels_cost(
                X_numerical,
                X_categorical,
                centroids,
                gamma,
                membership,
            )
            is_converged = (moves == 0) or (ncost >= cost)
            epoch_costs.append(ncost)
            cost = ncost
            if self.logging:
                print(
                    f"Run: {run_number + 1}, iteration: {iteration}/{self.max_iterations}, "
                    f"moves: {moves}, cost: {cost}"
                )

        return centroids, labels, cost, iteration, epoch_costs

    def __k_prototypes_iteration(
        self,
        X_numerical: np.ndarray,
        X_categorical: np.ndarray,
        centroids: np.ndarray | list,
        attributes_sums: np.ndarray,
        memberships_sums: np.ndarray,
        attributes_frequencies: np.ndarray | list,
        membership: np.ndarray,
        gamma: float,
        random_state: RandomState,
    ):
        moves = 0
        for point in range(X_numerical.shape[0]):
            weight = self.sample_weight[point] if self.sample_weight is not None else 1
            cluster = np.argmin(
                euclidean_dissimilarity(centroids[0], X_numerical[point])
                + gamma
                * matching_dissimilarity(
                    centroids[1],
                    X_categorical[point],
                    X=X_categorical,
                    membship=membership,
                )
            )
            if membership[cluster, point]:
                continue

            moves += 1
            old_cluster = np.argwhere(membership[:, point])[0][0]

            attributes_sums, memberships_sums = self._move_point_numerical(
                X_numerical[point],
                cluster,
                old_cluster,
                attributes_sums,
                memberships_sums,
                weight,
            )
            attributes_frequencies, membership, centroids[1] = (
                self._move_point_categorical(
                    X_categorical[point],
                    point,
                    cluster,
                    old_cluster,
                    attributes_frequencies,  # type: ignore
                    membership,
                    centroids[1],
                    weight,
                )
            )

            for i in range(len(X_numerical[point])):
                for j in (cluster, old_cluster):
                    if memberships_sums[j]:
                        centroids[0][j, i] = attributes_sums[j, i] / memberships_sums[j]
                    else:
                        centroids[0][j, i] = 0.0

            if not memberships_sums[old_cluster]:
                from_cluster = membership.sum(axis=1).argmax()
                choices = [
                    i for i, choice in enumerate(membership[from_cluster, :]) if choice
                ]
                random_index = random_state.choice(choices)

                attributes_sums, memberships_sums = self._move_point_numerical(
                    X_numerical[random_index],
                    old_cluster,
                    from_cluster,
                    attributes_sums,
                    memberships_sums,
                    weight,
                )
                attributes_frequencies, membership, centroids[1] = (
                    self._move_point_categorical(
                        X_categorical[random_index],
                        random_index,
                        old_cluster,
                        from_cluster,
                        attributes_frequencies,
                        membership,
                        centroids[1],
                        weight,
                    )
                )

        return (
            centroids,
            attributes_sums,
            memberships_sums,
            attributes_frequencies,
            membership,
            moves,
        )

    def __labels_cost(
        self,
        X_numerical: np.ndarray,
        X_categorical: np.ndarray,
        centroids: np.ndarray | list,
        gamma: float | None,
        membship=None,
        sample_weight=None,
    ):

        points_amount = X_numerical.shape[0]
        X_numerical = check_array(X_numerical)

        cost = 0.0
        labels = np.empty(points_amount, dtype=np.uint16)
        for i in range(points_amount):
            numerical = euclidean_dissimilarity(centroids[0], X_numerical[i])
            categorical = matching_dissimilarity(
                centroids[1], X_categorical[i], X=X_categorical, membship=membship
            )
            total_costs = numerical + gamma * categorical
            cluster = np.argmin(total_costs)
            labels[i] = cluster
            if self.sample_weight is not None:
                cost += total_costs[cluster] * self.sample_weight[i]
            else:
                cost += total_costs[cluster]

        return labels, cost

    def _move_point_numerical(
        self,
        point,
        to_cluster,
        from_cluster,
        attributes_sums,
        memberships_sums,
        sample_weight,
    ):
        for i, attribute in enumerate(point):
            attributes_sums[to_cluster][i] += attribute * sample_weight
            attributes_sums[from_cluster][i] -= attribute * sample_weight
        memberships_sums[to_cluster] += 1
        memberships_sums[from_cluster] -= 1
        return attributes_sums, memberships_sums

    def _split_num_cat(self, X, categorical):
        X_numerical = np.asanyarray(
            X[:, [i for i in range(X.shape[1]) if i not in categorical]]
        ).astype(np.float64)
        X_categorical = np.asanyarray(X[:, categorical])
        return X_numerical, X_categorical


def _validate_sample_weight(sample_weight, n_samples, n_clusters):
    if sample_weight is not None:
        if len(sample_weight) != n_samples:
            print(n_samples)
            raise ValueError("sample_weight should be of equal size as samples.")
        if any(
            not isinstance(weight, int) and not isinstance(weight, float)
            for weight in sample_weight
        ):
            raise ValueError("sample_weight elements should either be int or floats.")
        if any(sample < 0 for sample in sample_weight):
            raise ValueError("sample_weight elements should be positive.")
        if sum([x > 0 for x in sample_weight]) < n_clusters:
            raise ValueError(
                "Number of non-zero sample_weight elements should be "
                "larger than the number of clusters."
            )
