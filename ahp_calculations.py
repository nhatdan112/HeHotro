import numpy as np
import logging

# Configure logging to match routes.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_matrix(matrix):
    """Validate the input matrix for AHP calculations."""
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Matrix must be a NumPy array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}.")
    if np.any(matrix <= 0):
        raise ValueError("Matrix elements must be positive.")
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ValueError("Matrix contains NaN or infinite values.")
    logger.info(f"Validated matrix with shape {matrix.shape}")

def calculate_column_sums(matrix):
    """Calculate the sum of each column in the matrix."""
    col_sums = np.sum(matrix, axis=0)
    if np.any(col_sums == 0):
        raise ValueError("Column sums cannot be zero. Please revise your matrix.")
    logger.info(f"Column sums: {col_sums}")
    return col_sums

def normalize_matrix(matrix):
    """Normalize the matrix by dividing each element by its column sum."""
    try:
        col_sums = calculate_column_sums(matrix)
        normalized_matrix = matrix / col_sums
        logger.info(f"Normalized matrix:\n{normalized_matrix}")
        return normalized_matrix, col_sums
    except Exception as e:
        logger.error(f"Error normalizing matrix: {str(e)}")
        raise ValueError(f"Error normalizing matrix: {str(e)}")

def calculate_weights(normalized_matrix):
    """Calculate the weights by averaging each row of the normalized matrix."""
    try:
        weights = np.mean(normalized_matrix, axis=1)
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Weights contain NaN or infinite values.")
        logger.info(f"Weights: {weights}")
        return weights
    except Exception as e:
        logger.error(f"Error calculating weights: {str(e)}")
        raise ValueError(f"Error calculating weights: {str(e)}")

def calculate_weighted_sum_matrix(matrix, weights):
    """Calculate the weighted sum: original matrix * weights."""
    try:
        weighted_sum = matrix @ weights
        logger.info(f"Weighted sum: {weighted_sum}")
        return weighted_sum
    except Exception as e:
        logger.error(f"Error calculating weighted sum: {str(e)}")
        raise ValueError(f"Error calculating weighted sum: {str(e)}")

def calculate_consistency_vector(weighted_sum, weights):
    """Calculate the consistency vector: weighted sum / weights."""
    try:
        if np.any(weights == 0):
            raise ValueError("Weights cannot be zero. Please revise your matrix.")
        consistency_vector = weighted_sum / weights
        if np.any(np.isnan(consistency_vector)) or np.any(np.isinf(consistency_vector)):
            raise ValueError("Consistency vector contains NaN or infinite values.")
        logger.info(f"Consistency vector: {consistency_vector}")
        return consistency_vector
    except Exception as e:
        logger.error(f"Error calculating consistency vector: {str(e)}")
        raise ValueError(f"Error calculating consistency vector: {str(e)}")

def calculate_lambda_max(consistency_vector):
    """Calculate lambda_max: average of the consistency vector."""
    try:
        lambda_max = np.mean(consistency_vector)
        if np.isnan(lambda_max) or np.isinf(lambda_max):
            raise ValueError("Lambda_max is NaN or infinite.")
        logger.info(f"Lambda_max: {lambda_max}")
        return lambda_max
    except Exception as e:
        logger.error(f"Error calculating lambda_max: {str(e)}")
        raise ValueError(f"Error calculating lambda_max: {str(e)}")

def calculate_consistency_index(lambda_max, n):
    """Calculate the Consistency Index (CI)."""
    try:
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        logger.info(f"Consistency Index (CI): {ci}")
        return ci
    except Exception as e:
        logger.error(f"Error calculating CI: {str(e)}")
        raise ValueError(f"Error calculating CI: {str(e)}")

def calculate_consistency_ratio(ci, n):
    """Calculate the Consistency Ratio (CR)."""
    try:
        RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.54, 1.56, 1.57, 1.59]
        ri = RI[n-1] if n <= len(RI) else 1.59  # Use last RI for larger matrices
        cr = ci / ri if ri > 0 else 0
        logger.info(f"Consistency Ratio (CR): {cr}")
        return cr
    except Exception as e:
        logger.error(f"Error calculating CR: {str(e)}")
        raise ValueError(f"Error calculating CR: {str(e)}")

def normalize_tong_trong_so(weights):
    """Calculate and normalize the sum of weights."""
    try:
        tong_trong_so = np.sum(weights)
        if np.isnan(tong_trong_so) or np.isinf(tong_trong_so):
            raise ValueError("Sum of weights is NaN or infinite.")
        if not (0.95 <= tong_trong_so <= 1.05):
            raise ValueError(f"Sum of weights {tong_trong_so:.4f} is outside valid range (0.95–1.05).")
        logger.info(f"Sum of weights (tong_trong_so): {tong_trong_so}")
        return tong_trong_so
    except Exception as e:
        logger.error(f"Error normalizing tong_trong_so: {str(e)}")
        raise ValueError(f"Error normalizing tong_trong_so: {str(e)}")

def perform_ahp_steps(matrix):
    """Perform all AHP steps and return intermediate results."""
    try:
        # Validate input matrix
        validate_matrix(matrix)
        n = matrix.shape[0]
        logger.info(f"Starting AHP calculations for {n}x{n} matrix")

        # Step 1: Normalize the matrix
        normalized_matrix, col_sums = normalize_matrix(matrix)
        
        # Step 2: Calculate weights
        weights = calculate_weights(normalized_matrix)
        
        # Step 3: Calculate sum of weights
        tong_trong_so = normalize_tong_trong_so(weights)
        
        # Step 4: Calculate weighted sum
        weighted_sum = calculate_weighted_sum_matrix(matrix, weights)
        
        # Step 5: Calculate consistency vector
        consistency_vector = calculate_consistency_vector(weighted_sum, weights)
        
        # Step 6: Calculate lambda_max
        lambda_max = calculate_lambda_max(consistency_vector)
        
        # Step 7: Calculate CI
        ci = calculate_consistency_index(lambda_max, n)
        
        # Step 8: Calculate CR
        cr = calculate_consistency_ratio(ci, n)
        
        result = {
            "ma_tran": matrix.tolist(),
            "trong_so": weights.tolist(),
            "tong_cot": col_sums.tolist(),
            "ma_tran_chuan_hoa": normalized_matrix.tolist(),
            "tong_trong_so": float(tong_trong_so),  # Scalar for sum of weights
            "vector_nhat_quan": consistency_vector.tolist(),
            "lambda_max": float(lambda_max),
            "ci": float(ci),
            "cr": float(cr)
        }
        logger.info(f"AHP calculations completed: CR={cr:.4f}, tong_trong_so={tong_trong_so:.4f}")
        return result
    except Exception as e:
        logger.error(f"Error in AHP calculation: {str(e)}")
        raise ValueError(f"Error in AHP calculation: {str(e)}")