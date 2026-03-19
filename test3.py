import numpy as np

def saaty_consistency_check(matrix):
    n = matrix.shape[0]
    
    # Для матриц 1x1 и 2x2 индекс согласованности всегда равен 0
    if n <= 2:
        if n == 1:
            return True, 0.0, np.array([1.0]), None
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        w = np.abs(np.real(eigenvecs[:, np.argmax(np.real(eigenvals))]))
        w = w / np.sum(w)
        return True, 0.0, w, None
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenval = float(np.max(np.real(eigenvalues)))
    
    weights = np.abs(np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))]))
    weights = weights / np.sum(weights)
    
    ci = (max_eigenval - n) / (n - 1)
    
    # Табличные значения случайного индекса (RI) по Саати
    ri_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict.get(n, 1.49)
    
    cr = ci / ri
    is_consistent = cr <= 0.1
    
    problem_pair = None
    if not is_consistent:
        ideal_matrix = np.outer(weights, 1 / weights)
        error_matrix = np.maximum(matrix / ideal_matrix, ideal_matrix / matrix)
        error_matrix[np.tril_indices(n)] = 0
        
        # ADDED PRINT HERE
        print(f"n={n}")
        print(f"matrix=\n{matrix}")
        print(f"ideal_matrix=\n{ideal_matrix}")
        print(f"error_matrix=\n{error_matrix}")
        
        i_idx, j_idx = np.unravel_index(np.argmax(error_matrix), error_matrix.shape)
        problem_pair = (int(i_idx), int(j_idx))
    
    return is_consistent, cr, weights, problem_pair


m = np.array([
    [1, 9, 9],
    [1/9, 1, 9],
    [1/9, 1/9, 1]
])

saaty_consistency_check(m)
