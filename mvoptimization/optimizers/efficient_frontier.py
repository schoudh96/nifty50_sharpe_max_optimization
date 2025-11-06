import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

from . import base_optimizer

class EfficientFrontier(base_optimizer.BaseConvexOptimizer):

    """
    Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the portfolio for which the capital market line is tangent to the efficient frontier.

        This is a convex optimization problem after making a certain variable substitution. See
        `Cornuejols and Tutuncu (2006) <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>`_ for more.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: OrderedDict
    
    """

    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for volatility only (but not recommended).
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset. This **must** be
                           positive semidefinite, otherwise optimization will fail.
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`
        :type solver: str
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        :raises TypeError: if ``cov_matrix`` is not a dataframe or array
        """
        # Inputs
        self.cov_matrix = EfficientFrontier._validate_cov_matrix(cov_matrix)
        self.expected_returns = EfficientFrontier._validate_expected_returns(
            expected_returns
        )
        self._max_return_value = None
        self._market_neutral = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # use integer labels
            tickers = list(range(num_assets))

        if expected_returns is not None and cov_matrix is not None:
            if cov_matrix.shape != (num_assets, num_assets):
                raise ValueError("Covariance matrix does not match expected returns")

        super().__init__(
            len(tickers),
            tickers,
            weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if cov_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError("cov_matrix is not a dataframe or array")

    def _validate_returns(self, returns):
        """
        Helper method to validate daily returns (needed for some efficient frontiers)
        """
        if not isinstance(returns, (pd.DataFrame, np.ndarray)):
            raise TypeError("returns should be a pd.Dataframe or np.ndarray")

        returns_df = pd.DataFrame(returns)
        if returns_df.isnull().values.any():
            warnings.warn(
                "Removing NaNs from returns",
                UserWarning,
            )
            returns_df = returns_df.dropna(axis=0, how="any")

        if self.expected_returns is not None:
            if returns_df.shape[1] != len(self.expected_returns):
                raise ValueError(
                    "returns columns do not match expected_returns. Please check your tickers."
                )

        return returns_df

    def create_penalized_clusters(
    self,
    returns,
    n_clusters: int = 10,
    set_optimal_k=None,
    min_cluster_size: int = 4,
    lambda_pen: float = 10.0,
    ):
        """
        Same as create_clusters but uses a penalized K-means so clusters
        with size < min_cluster_size are discouraged.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        import numpy as np
        import matplotlib.pyplot as plt

        # ---------- helper: penalized kmeans for a fixed k ----------
        def _penalized_kmeans_fit(X, k, min_cluster_size, lambda_pen,
                                max_iter=100, random_state=42):
            rng = np.random.RandomState(random_state)
            n_samples, n_features = X.shape

            # ---- k-means++-ish init (fixed) ----
            centers = [X[rng.randint(0, n_samples)]]
            for _ in range(1, k):
                # distance of every point to its nearest existing center
                # shape after stack: (n_samples, current_num_centers)
                dist_mat = np.stack(
                    [np.sum((X - c) ** 2, axis=1) for c in centers],
                    axis=1
                )                        # (n_samples, m)
                dists = np.min(dist_mat, axis=1)  # (n_samples,)
                total = dists.sum()
                if total == 0:  # all points identical to a center
                    idx = rng.randint(0, n_samples)
                else:
                    probs = dists / total
                    idx = rng.choice(n_samples, p=probs)
                centers.append(X[idx])
            centers = np.stack(centers, axis=0)   # (k, n_features)

            labels = np.zeros(n_samples, dtype=int)

            for _ in range(max_iter):
                # distances to centers: (n_samples, k)
                dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

                # current cluster sizes
                cluster_sizes = np.bincount(labels, minlength=k)

                # add penalty to small clusters
                penalized_dists = dists.copy()
                for c in range(k):
                    if cluster_sizes[c] < min_cluster_size:
                        penalized_dists[:, c] += lambda_pen * (min_cluster_size - cluster_sizes[c])

                new_labels = np.argmin(penalized_dists, axis=1)

                if np.all(new_labels == labels):
                    labels = new_labels
                    break

                labels = new_labels

                # update centers
                for c in range(k):
                    members = X[labels == c]
                    if len(members) > 0:
                        centers[c] = members.mean(axis=0)
                    else:
                        # reinit empty cluster to random point
                        centers[c] = X[rng.randint(0, n_samples)]

            # compute inertia (SSE)
            inertia = 0.0
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    inertia += np.sum((members - centers[c]) ** 2)

            return labels, centers, inertia

        # ---------- your original flow ----------
        returns_df = self._validate_returns(returns)
        returns_df = returns_df.T  # assets as rows

        scaler = StandardScaler()
        X = scaler.fit_transform(returns_df)

        silhouette_scores = []
        wcss_values = []
        k_range = range(2, n_clusters + 1)

        print(f"Shape of returns matrix: {X.shape}")

        labels_per_k = {}
        for k in k_range:
            labels_k, centers_k, inertia_k = _penalized_kmeans_fit(
                X,
                k,
                min_cluster_size=min_cluster_size,
                lambda_pen=lambda_pen,
                max_iter=100,
                random_state=42,
            )
            # if all labels collapsed to 1 cluster, silhouette will fail,
            # so guard it
            if len(np.unique(labels_k)) > 1:
                score = silhouette_score(X, labels_k)
            else:
                score = -1.0  # bad clustering
            silhouette_scores.append(score)
            wcss_values.append(inertia_k)
            labels_per_k[k] = labels_k

        print(f"WCSS values for k=2 to {n_clusters}: {wcss_values}")
        print(f"First diff WCSS values for k=2 to {n_clusters}: {np.diff(wcss_values)}")

        plt.plot(k_range, wcss_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Penalized KMeans: WCSS vs k')
        plt.show()

        # pick best k by silhouette, unless overridden
        optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

        if set_optimal_k:
            optimal_k = set_optimal_k

        print(f"Using optimal k = {optimal_k} for clustering.")

        final_labels = labels_per_k[optimal_k]

        # build clusters
        self.clusters = [[] for _ in range(optimal_k)]
        for idx, label in enumerate(final_labels):
            self.clusters[label].append(idx)

        # drop empties if any
        self.clusters = [c for c in self.clusters if len(c) > 0]

        return self.clusters



    def create_clusters(self, returns, n_clusters: int = 10, set_optimal_k = None):
        """
        Create clusters of assets based on their return correlations.

        :param returns: daily returns of the assets
        :param n_clusters: number of clusters to create

        :returns: list of clusters, each containing asset indices
        :rtype: list of lists
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn import metrics
        from scipy.spatial.distance import cdist
        from k_means_constrained import KMeansConstrained

        returns_df = self._validate_returns(returns)
        returns_df = returns_df.T  # Transpose to have assets as rows

        # Calculate correlation matrix
        # corr_matrix = returns_df.corr().values

        # Standardize the correlation matrix
        scaler = StandardScaler()
        returns_matrix_std = scaler.fit_transform(returns_df)

        # Apply KMeans clustering
        silhouette_scores = []
        wcss_values = []
        k_range = range(2, n_clusters + 1)

        print(f'Shape of returns matrix: {returns_matrix_std.shape}')

        for k in k_range:
            # kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans = KMeansConstrained(n_clusters=k, random_state=42, size_min = 3)
            kmeans.fit(returns_matrix_std)
            score = silhouette_score(returns_matrix_std, kmeans.labels_)
            silhouette_scores.append(score)
            wcss_values.append(kmeans.inertia_)

        print(f'WCSS values for k=2 to {n_clusters}: {wcss_values}')
        print(f'First diff WCSS values for k=2 to {n_clusters}: {np.diff(wcss_values)}')
        import matplotlib.pyplot as plt
        plt.plot(k_range, wcss_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal k')
        plt.show()
        # Choose the optimal number of clusters (elbow method)
        # optimal_k = n_clusters  # Default to n_clusters if elbow not found
        # if n_clusters >= 3:
        #     deltas = np.diff(wcss_values)
        #     second_deltas = np.diff(deltas)
        #     optimal_k = np.argmin(second_deltas) + 2  # +2 due to the nature of diff
        # kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        # kmeans.fit(returns_matrix_std)

        optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        print(f'Optimal number of clusters based on silhouette score: {optimal_k}')

        if set_optimal_k:
            optimal_k = set_optimal_k

        print(f"Using optimal k = {optimal_k} for clustering.")

        # kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans = KMeansConstrained(n_clusters=optimal_k, random_state=42, size_min = 3)
        kmeans.fit(returns_matrix_std)

        self.clusters = [[] for _ in range(optimal_k)]
        for idx, label in enumerate(kmeans.labels_):
            self.clusters[label].append(idx)

        return self.clusters

    def max_sharpe(self, risk_free_rate=0.0, needed_nco = False):
        """
        Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the portfolio for which the capital market line is tangent to the efficient frontier.

        This is a convex optimization problem after making a certain variable substitution. See
        `Cornuejols and Tutuncu (2006) <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>`_ for more.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: OrderedDict
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        if max(self.expected_returns) <= risk_free_rate:
            raise ValueError(
                "at least one of the assets must have an expected return exceeding the risk-free rate"
            )

        self._risk_free_rate = risk_free_rate

        # max_sharpe requires us to make a variable transformation.
        # Here we treat w as the transformed variable.
        self._objective = cp.quad_form(self._w, self.cov_matrix)
        k = cp.Variable()

        # Note: objectives are not scaled by k. Hence there are subtle differences
        # between how these objectives work for max_sharpe vs min_volatility
        if len(self._additional_objectives) > 0:
            warnings.warn(
                "max_sharpe transforms the optimization problem so additional objectives may not work as expected."
            )
        for obj in self._additional_objectives:
            self._objective += obj

        new_constraints = []
        # Must rebuild the constraints
        for constr in self._constraints:
            if isinstance(constr, cp.constraints.nonpos.Inequality):
                # Either the first or second item is the expression
                if isinstance(
                    constr.args[0], cp.expressions.constants.constant.Constant
                ):
                    new_constraints.append(constr.args[1] >= constr.args[0] * k)
                else:
                    new_constraints.append(constr.args[0] <= constr.args[1] * k)
            elif isinstance(constr, cp.constraints.zero.Equality):
                new_constraints.append(constr.args[0] == constr.args[1] * k)
            else:
                raise TypeError(
                    "Please check that your constraints are in a suitable format"
                )

        # Transformed max_sharpe convex problem:
        self._constraints = [
            (self.expected_returns - risk_free_rate).T @ self._w == 1,
            cp.sum(self._w) == k,
            k >= 0,
        ] + new_constraints

        # self._constraints = [
        #     (self.expected_returns - risk_free_rate).T @ self._w == 1,
        #     cp.sum(self._w) == 1,
        #     self._w >= 0,
        # ] + new_constraints

        # self.weights = (self._w.value).round(16) + 0.0

        self._solve_cvxpy_opt_problem()
        # Inverse-transform
        # import pdb
        # pdb.set_trace()
        # print(f'Weights = {self._w}')
        # # self.weights = self._w
        # self.weights = (self._w.value).round(16) + 0.0
        print(self._w.value)
        print(f'Sum weights: {sum(self._w.value)}')
        self.weights = (self._w.value / k.value).round(16) + 0.0

        if needed_nco:
            return self.weights
        return self._make_output_weights()

    def nco_max_sharpe(self, returns = None, clusters = None, weight_bounds_generic = None, risk_free_rate=0.0):
        """
        Run Nested Clustered Optimization (NCO) to maximise the Sharpe Ratio.
        """
        # Step 1: Create clusters of similar assets
        # clusters = self.create_clusters(returns, n_clusters=5)

        # Step 2: Optimize each cluster separately
        cluster_weights = []
        clusters_mean_exp_returns = []
        cluster_expected_returns = pd.DataFrame()
        cluster_weights_df = []

        # import pdb 
        # pdb.set_trace()
        for c_idx, cluster in enumerate(clusters):
            #carve out the expected returns and covariance matrix for the cluster from self.expected_returns and self.cov_matrix
            cluster_mean_returns = self.expected_returns[cluster]
            cluster_cov_matrix = pd.DataFrame(self.cov_matrix).iloc[cluster, cluster].values

            print(f'\n\n for cluster = {c_idx}, len of cluster = {cluster} and items in cluster = {cluster}')
            #run intra cluster optimization

            # if len(cluster) <= 3:
            #     intra_bounds = (0.0, 1.0)
            # else:
            #     # fall back to user generic bounds or something mild
            #     intra_bounds = weight_bounds_generic

            if len(cluster) <= 5:
                ef_intracluster = EfficientFrontier(cluster_mean_returns, cluster_cov_matrix, weight_bounds=(0,1))
                weights_intracluster = ef_intracluster.max_sharpe(risk_free_rate=risk_free_rate, needed_nco = True)

            else:
                ef_intracluster = EfficientFrontier(cluster_mean_returns, cluster_cov_matrix, weight_bounds=(0.002*len(clusters),1))
                weights_intracluster = ef_intracluster.max_sharpe(risk_free_rate=risk_free_rate, needed_nco = True)

            print(f"\n\n For cluster {c_idx}; len weights vector = {len(weights_intracluster)}; sum weights vector = {sum(weights_intracluster)}; min-max of weights vector = {(min(weights_intracluster), max(weights_intracluster))}")
            #create dataframe of intra cluster weights
            print(f'Tickers selected for cluster {c_idx}: {np.array(self.tickers)[cluster]}')
            cluster_weights_df.append(pd.DataFrame(zip(np.array(self.tickers)[cluster], list(weights_intracluster)), columns = ['ticker', 'weight']))
            
            print(f'\nShape of intra cluster weights df = {cluster_weights_df[c_idx].shape}')
            #extract only weights
            weights_intracluster = list(weights_intracluster)
            cluster_weights.append(weights_intracluster)


            cluster_returns = returns.iloc[:, cluster].values @ weights_intracluster

            cluster_length = len(cluster)
            return_multiplier = 1/(1 - np.log(1 + cluster_length/50))
            cluster_expected_returns[c_idx] = cluster_returns*return_multiplier

            print(f'For cluster with assets size = {len(cluster)}; multiplier = {return_multiplier}; cluster mean return = {np.dot(weights_intracluster, cluster_mean_returns)}')
            clusters_mean_exp_returns.append(np.dot(weights_intracluster, cluster_mean_returns))

        
        # using cluster weights create a new expected returns and covariance matrix for the clusters
        intracluster_cov_matrix = cluster_expected_returns.cov()

        print(f'to check correlation of interclusters = {cluster_expected_returns.corr()}')
        # plt.heatmap(cluster_expected_returns.corr(), annot=True)

        ef_intercluster = EfficientFrontier(clusters_mean_exp_returns, intracluster_cov_matrix, weight_bounds=(0.1, 0.5))
        weights_intercluster = ef_intercluster.max_sharpe(risk_free_rate=risk_free_rate)

        final_weights = pd.DataFrame()
        for c_idx, cluster in enumerate(clusters):
            cluster_weights_df[c_idx]['weight'] = cluster_weights_df[c_idx]['weight'] * weights_intercluster[c_idx]
            final_weights = final_weights.append(cluster_weights_df[c_idx])

        return final_weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calculated yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        if self._risk_free_rate is not None:
            if risk_free_rate != self._risk_free_rate:
                warnings.warn(
                    "The risk_free_rate provided to portfolio_performance is different"
                    " to the one used by max_sharpe. Using the previous value.",
                    UserWarning,
                )
            risk_free_rate = self._risk_free_rate

        return base_optimizer.portfolio_performance(
            self.weights,
            self.expected_returns,
            self.cov_matrix,
            verbose,
            risk_free_rate,
        )