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