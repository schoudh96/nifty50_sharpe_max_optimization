import pandas as pd
import numpy as np
import os

class CovarianceEstimator:
    def __init__(self, pricedata: pd.DataFrame):
        self.returns = pricedata.pct_change().dropna()

    def sample_covariance(self, returns_df, frequency: int = 252) -> pd.DataFrame:
        """
        Calculate the annual sample covariance matrix of daily asset returns.

        :param frequency: days to use for annualization

        :returns annualised sample covariance matrix
        :rtype pd.DataFrame
        """
        cov_matrix = returns_df.cov() * frequency
        return cov_matrix
    
    def helper_numberfactors(self, evals: np.ndarray, var_explained: float = 0.9, rank: int = None) -> int:
        """
        Helper function to determine number of factors based on explained variance or fixed rank.

        :param evals: Eigenvalues of the covariance matrix
        :param var_explained: Target variance to be explained by factors
        :param rank: Fixed number of factors

        :returns number of factors to use
        :rtype int
        """
        if rank is not None:
            return rank
        
        total_variance = np.sum(evals)
        explained_variance = 0.0
        k_use = 0

        for k, val in enumerate(evals):
            explained_variance += val
            if explained_variance / total_variance >= var_explained:
                k_use = k + 1
                break
        
        return k_use
    
    def factor_analysis_shared_covariance(self, covar_matrix, ridge_penalty = False, cutoff_eigen_cum_ratio = 0.9, run_multiple_times = False, run_count = 5):

        """
        Run factor analysis on shared covariance and complete PCA-based denoising.
        Smooth specific variances so as to improve estimation.
        If ridge_penalty is True, apply ridge penalty to specific variances to improve inversion stability. : not implemented yet

        :param covar_matrix: Covariance matrix to denoise
        """
        #convert covar_matrix into correlation matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(covar_matrix)))
        corr_matrix = D_inv_sqrt @ covar_matrix @ D_inv_sqrt


        evals, evecs = np.linalg.eigh(corr_matrix)
        # Sort descending: largest eigenvalues first
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        #get number of factors to retain
        k_use = self.helper_numberfactors(evals, var_explained=cutoff_eigen_cum_ratio, rank=None)

        # Build shared covariance from top-k eigenmodes
        Vk = evecs[:, :k_use]
        Lk = np.diag(evals[:k_use])
        corr_shared = Vk @ Lk @ Vk.T

        # Specific variances from residual diagonal (preserve total variance)
        diag_corr = np.diag(corr_matrix)
        diag_shared = np.diag(corr_shared)
        psi = diag_corr - diag_shared

        if run_multiple_times:
            count = 0
            while count <= run_count:
                count += 1
                corr_shared_iter = corr_matrix - np.diag(psi) 
                evals, evecs = np.linalg.eigh(corr_shared_iter)
                # Sort descending: largest eigenvalues first
                idx = np.argsort(evals)[::-1]
                evals = evals[idx]
                evecs = evecs[:, idx]

                #get number of factors to retain
                # k_use = self.helper_numberfactors(evals, var_explained=cutoff_eigen_cum_ratio, rank=None)

                # Build shared covariance from top-k eigenmodes
                Vk = evecs[:, :k_use]
                Lk = np.diag(evals[:k_use])
                corr_shared = Vk @ Lk @ Vk.T

                # Specific variances from residual diagonal (preserve total variance)
                diag_corr = np.diag(corr_matrix)
                diag_shared = np.diag(corr_shared)
                psi = diag_corr - diag_shared

        print(f"\n Evals total = {evals}")
        print(f'\n evalues used for factors: {np.diag(Lk)}')
        
        print(f'\ncorr matrix diag: {diag_corr}')
        print(f'\ncorr shared diag: {diag_shared}')
        print(f"\n\npsi before setting as mean = {psi}")
        # psi[k_use:] = np.mean(psi[k_use:])

        print(f'\n\nafter setting low psi to mean : {psi}')
        print(f"Number of factors used: {k_use} out of {len(evals)}")
        
        print(f"Min/Max specific variance before cov scaling: {np.min(psi)}/{np.max(psi)}")

        D = np.diag(np.sqrt(np.diag(covar_matrix)))
        cov_shared = D @ corr_shared @ D
        cov_unique = D @ np.diag(psi) @ D

        print(f"\nCov unique diag: {(np.diag(cov_unique))}")
        # print(f"Std dev matrix: {D}")
        print(f"Min/Max specific variance after scaling to covariance: {np.min(cov_unique)}/{np.max(cov_unique)}")

        print(f'cov shared matrix saved to path: {os.path.join(os.getcwd(), "cov_shared.csv")}')
        pd.DataFrame(cov_shared).to_csv(os.path.join(os.getcwd(), "cov_shared.csv"))
        
        print(f'cov unique matrix saved to path: {os.path.join(os.getcwd(), "cov_unique.csv")}')
        pd.DataFrame(cov_unique).to_csv(os.path.join(os.getcwd(), "cov_unique.csv"))

        estimated_total_cov = cov_unique + cov_shared  
        return estimated_total_cov

    def run_oos_testing(self, cutoff_eigen_ratio = 0.9, run_multiple_times = False, test_unique = False):
        """
        Run out-of-sample testing to evaluate covariance estimation efficacy.
        """
        time_idx = self.returns.index
        est_cov_errors = []
        sample_cov_errors = []
        for t in range(252, len(time_idx)-253, 20):
            train_returns = self.returns.iloc[t-252:t]
            test_returns = self.returns.iloc[t:t+252]

            # Estimate covariance on training data
            train_cov_matrix = self.sample_covariance(train_returns)
            est_cov_matrix = self.factor_analysis_shared_covariance(train_cov_matrix, cutoff_eigen_cum_ratio=cutoff_eigen_ratio, run_multiple_times=run_multiple_times)
            
            # Sample covariance on test data
            test_cov_matrix = self.sample_covariance(test_returns)

            print(f'\n Total sample covariance error against oos = {np.linalg.norm(train_cov_matrix - test_cov_matrix, ord="fro")}')
            print(f'\n Total estimated covariance error against oos = {np.linalg.norm(est_cov_matrix - test_cov_matrix, ord="fro")}')
            if test_unique:
                
                # Compute frobenius error metric between estimated and test covariance
                est_cov_error = np.linalg.norm((np.diag(np.diag(est_cov_matrix))) - np.diag(np.diag(test_cov_matrix)), ord='fro')
                
                #compute sample covariance error for comparison
                sample_cov_error = np.linalg.norm((np.diag(np.diag(train_cov_matrix))) - np.diag(np.diag(test_cov_matrix)), ord='fro')
            
            else:
                # Compute frobenius error metric between estimated and test covariance
                est_cov_error = np.linalg.norm((est_cov_matrix - np.diag(est_cov_matrix)) - (test_cov_matrix - np.diag(test_cov_matrix)), ord='fro')
                
                #compute sample covariance error for comparison
                sample_cov_error = np.linalg.norm((train_cov_matrix - np.diag(train_cov_matrix)) - (test_cov_matrix - np.diag(test_cov_matrix)), ord='fro')
            

            print(f"\n\nest cov matrix: {np.linalg.norm(est_cov_matrix, ord='fro')}")
            print(f"test cov matrix: {np.linalg.norm(test_cov_matrix, ord='fro')}")
            print(f"Time {time_idx[t]}: Covariance estimation error (Frobenius norm): {est_cov_error}\n")

            est_cov_errors.append(est_cov_error)
            sample_cov_errors.append(sample_cov_error)

        avg_esterror = np.mean(est_cov_errors)
        avg_sampleerror = np.mean(sample_cov_errors)
        print(f"\nAverage covariance estimation error over OOS tests: {avg_esterror}")
        print(f"Average sample covariance error over OOS tests: {avg_sampleerror}\n\n")
        
        return (avg_esterror, avg_sampleerror)