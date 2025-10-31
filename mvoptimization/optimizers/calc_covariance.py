import pandas as pd
import numpy as np
import warnings

def _is_positive_semidefinite(matrix):
    """
    Helper function to check if a given matrix is positive semidefinite.
    Any method that requires inverting the covariance matrix will struggle
    with a non-positive semidefinite matrix

    :param matrix: (covariance) matrix to test
    :type matrix: np.ndarray, pd.DataFrame
    :return: whether matrix is positive semidefinite
    :rtype: bool
    """
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False

def fix_nonpositive_semidefinite(matrix, fix_method = 'spectral'):
    
    warnings.warn('The eigenvalues are non positive, i.e. non positive semidefinite. Amending eigenvalues.')

    q, V = np.linalg.eigh(matrix)

    if fix_method == 'spectral':
        #remove negative eigenvalues
        q = np.where(q<0, 0, q)
        fixed_matrix = V @ np.diag(q) @ V.T

    if not _is_positive_semidefinite:
        raise Exception('Could not fix covariance matrix.')
    
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix



def sample_correlation(returns:pd.DataFrame, frequency:int = 252) -> pd.DataFrame:
    """
    Calculate the annual sample correlation matrix of daily asset returns.

    :param returns: returns of the adjusted price series
    :param frequency: days to use for annualization

    :returns annualised sample correlation matrix
    :rtype pd.DataFrame
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f'Returns DataFrame has to be a pandas DataFrame.Type provided is {type(returns)}.')
    
    corr_matrix = returns.corr() # * frequency

    if _is_positive_semidefinite(corr_matrix):
        return corr_matrix
    
    else:
        corr_matrix = fix_nonpositive_semidefinite(corr_matrix)
        return corr_matrix
    


def sample_covariance(returns:pd.DataFrame, frequency:int = 252) -> pd.DataFrame:
    """
    Calculate the annual sample covariance matrix of daily asset returns.

    :param returns: returns of the adjusted price series
    :param frequency: days to use for annualization

    :returns annualised sample covariance matrix
    :rtype pd.DataFrame
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f'Returns DataFrame has to be a pandas DataFrame.Type provided is {type(returns)}.')
    
    cov_matrix = returns.cov() * frequency

    if _is_positive_semidefinite(cov_matrix):
        return cov_matrix
    
    else:
        cov_matrix = fix_nonpositive_semidefinite(cov_matrix)
        return cov_matrix
    

def exponential_covariance(returns:pd.DataFrame, span:int, frequency:int = 252) -> pd.DataFrame:
    """
    Calculate the annual exponential covariance matrix of daily asset returns.

    :param returns: returns of the adjusted price series
    :param frequency: days to use for annualization

    :returns annualised sample covariance matrix
    :rtype pd.DataFrame
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f'Returns DataFrame has to be a pandas DataFrame.Type provided is {type(returns)}.')
    
    cov_matrix = pd.ewmcov(returns, span = span) * frequency

    if _is_positive_semidefinite(cov_matrix):
        return cov_matrix
    
    else:
        cov_matrix = fix_nonpositive_semidefinite(cov_matrix)
        return cov_matrix
    


def denoising_covariance(returns:pd.DataFrame, covar_matrix: pd.DataFrame, frequency:int = 252, rank: int = None,
    alpha: float = 0.0,
    var_explained: float = None,
    ensure_spd: bool = True,
    eps: float = 1e-10) -> pd.DataFrame:
    """
    Calculate the annual denoised covariance matrix of daily asset returns.

    :param returns: returns of the adjusted price series
    :param frequency: days to use for annualization

    :returns annualised denoised covariance matrix
    :rtype pd.DataFrame
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f'Returns DataFrame has to be a pandas DataFrame.Type provided is {type(returns)}.')
    
    # from mvoptimization.optimizers.covariance_factorization import low_rank_covariance


    # cov_matrix = returns.cov().values
    # denoised_cov_matrix = low_rank_covariance(returns.values)

    # denoised_cov_matrix = pd.DataFrame(denoised_cov_matrix * frequency, index=returns.columns, columns=returns.columns)

    # if _is_positive_semidefinite(denoised_cov_matrix):
    #     return denoised_cov_matrix
    
    # else:
    #     denoised_cov_matrix = fix_nonpositive_semidefinite(denoised_cov_matrix)
    #     return denoised_cov_matrix

    if alpha < 0:
        raise ValueError("alpha (ridge penalty) must be >= 0.")

    # # Align: drop rows with any NaNs so sample cov is well-defined
    # R = returns.dropna(how="any").copy()
    # cols = R.columns.tolist()
    # n = len(cols)
    # if n == 0 or R.shape[0] < 2:
    #     raise ValueError("Not enough clean data to compute covariance.")

    # # 1) Daily covariance
    # if covar_matrix is None:
    #     Sigma = np.cov(R.values, rowvar=False, ddof=1)
    # else:
    #     # Ensure ordering matches 'returns' columns
    #     Sigma = covar_matrix.loc[cols, cols].values.astype(float, copy=True)


    #spectral decomposition 
    evals, evecs = np.linalg.eigh(covar_matrix)
    # sort eigenvalues in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    print("Eigenvalues:", evals)
    print(f"Ratio of top to bottom eigenvalue: evals[0]/evals[-1] = {evals[0]/evals[-1]}")

    return None

    # # 3) Choose rank if requested
    # if rank is None and var_explained is not None:
    #     if not (0.0 < var_explained <= 1.0):
    #         raise ValueError("var_explained must be in (0, 1].")
    #     cum = np.cumsum(evals.clip(min=0.0))
    #     total = cum[-1] if cum[-1] > 0 else 0.0
    #     if total == 0.0:
    #         # Degenerate case: all-zero covariance
    #         rank = 0
    #     else:
    #         rank = int(np.searchsorted(cum / total, var_explained) + 1)

    # # Bound rank
    # if rank is None:
    #     rank = n  # full rank
    # rank = max(0, min(int(rank), n))

    # # 4) Low-rank reconstruction via top 'rank' PCs
    # if rank == 0:
    #     Sigma_lr = np.zeros((n, n), dtype=float)
    # elif rank == n:
    #     # Full reconstruction equals original Σ (before ridge)
    #     Sigma_lr = (evecs * evals) @ evecs.T
    # else:
    #     V_r = evecs[:, :rank]
    #     Λ_r = np.diag(evals[:rank])
    #     Sigma_lr = V_r @ Λ_r @ V_r.T

    # # 5) Ridge penalty: add alpha * I
    # if alpha > 0:
    #     Sigma_ridge = Sigma_lr + alpha * np.eye(n, dtype=float)
    # else:
    #     Sigma_ridge = Sigma_lr

    # # 6) Ensure SPD (optional)
    # if ensure_spd:
    #     d, U = np.linalg.eigh(Sigma_ridge)
    #     d = np.maximum(d, eps)
    #     Sigma_spd = (U * d) @ U.T
    # else:
    #     Sigma_spd = Sigma_ridge

    # # 7) Annualize
    # Sigma_ann = frequency * Sigma_spd

    # return pd.DataFrame(Sigma_ann, index=cols, columns=cols)