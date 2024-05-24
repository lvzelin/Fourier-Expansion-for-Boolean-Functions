import numpy as np
from itertools import product

# random read-k permutation branching program
def random_perm_read_k_BP(n,k,w):
    # Create an array with each element from 1 to n appearing k times
    var_ordering = np.tile(np.arange(1, n+1), k)
    # Shuffle the var_ordering to randomize the order of elements
    np.random.shuffle(var_ordering)
    print(var_ordering)
    permutation_matrices_0 = []
    permutation_matrices_1 = []
    for _ in range(n*k):
        perm = np.random.permutation(w)
        perm_matrix = np.zeros((w, w))
        perm_matrix[np.arange(w), perm] = 1
        permutation_matrices_0.append(perm_matrix)

        perm = np.random.permutation(w)
        perm_matrix = np.zeros((w, w))
        perm_matrix[np.arange(w), perm] = 1
        permutation_matrices_1.append(perm_matrix)

    for i, matrix in enumerate(permutation_matrices_0):
        print(f"Positive Permutation Matrix {i+1}:\n{matrix}\n")
    
    for i, matrix in enumerate(permutation_matrices_1):
        print(f"negative Permutation Matrix {i+1}:\n{matrix}\n")

    num_rows = 2 ** n
    truth_table = np.zeros((num_rows, n + 1))

    for i in range(num_rows):
        assignment = np.array(list(map(int, np.binary_repr(i, width=n))))
        current_matrix = np.eye(w)
        
        for j in range(n * k):
            var_index = var_ordering[j] - 1  # Adjust for 0-based indexing in Python
            if assignment[var_index] == 1:
                current_matrix = np.dot(current_matrix, permutation_matrices_1[j])
            else:
                current_matrix = np.dot(current_matrix, permutation_matrices_0[j])
        
        # Assign the (1,1)-element of the final matrix to the function value
        truth_table[i, n] = current_matrix[0, 0]
        truth_table[i, :n] = assignment

    return truth_table


def mod_2k(n, k):
    num_rows = 2 ** n
    truth_table = np.zeros((num_rows, n + 1))

    for i in range(num_rows):
        truth_table[i, :n] = np.array(list(map(int, np.binary_repr(i, width=n)))) * 2 - 1
    
    num_neg1 = truth_table[:, :n] - 1
    num_neg1 = np.abs(np.sum(num_neg1, axis=1) / 2)
    num_neg1 = np.mod(num_neg1, 2 * k)

    truth_table[num_neg1 <= k - 1, n] = -1
    truth_table[num_neg1 > k - 1, n] = 1
    
    return truth_table

def or_function(n):
    num_rows = 2 ** n
    truth_table = np.zeros((num_rows, n + 1))

    for i in range(num_rows):
        truth_table[i, :n] = np.array(list(map(int, np.binary_repr(i, width=n))))
    
    truth_table[:, n] = np.any(truth_table[:, :n], axis=1)
    return truth_table

def majority_function(n):
    num_rows = 2 ** n
    truth_table = np.zeros((num_rows, n + 1))

    for i in range(num_rows):
        truth_table[i, :n] = np.array(list(map(int, np.binary_repr(i, width=n)))) * 2 - 1
    
    truth_table[:, n] = np.sign(np.sum(truth_table[:, :n], axis=1))
    truth_table[truth_table[:, n] == 0, n] = 1
    return truth_table

def boolean_fourier_expansion(f):
    n = int(np.log2(len(f)))
    fourier_expansion = [None] * len(f)
    coeffs = np.zeros(len(f))

    truth_table = np.array(list(product([-1, 1], repeat=n)))

    for i in range(len(f)):
        binary = np.array(list(map(int, np.binary_repr(i, width=n))))
        character_table = np.prod(truth_table[:, binary == 1], axis=1)
        coeff = np.dot(f, character_table) / len(f)
        coeffs[i] = coeff

        if np.any(binary):
            str_exp = ''
            for j in range(n):
                if binary[j] == 1:
                    str_exp += f'x{j + 1}'
            if coeff < 0:
                str_exp = f'-{abs(coeff)} {str_exp} '
            else:
                str_exp = f'+{coeff} {str_exp} '
            fourier_expansion[i] = str_exp
    
    fourier_expansion = [exp for exp in fourier_expansion if exp]
    return coeffs, fourier_expansion

def fourier_coeffs_to_boolean_function(coeffs):
    n = int(np.log2(len(coeffs)))
    num_inputs = 2 ** n
    truth_table = np.zeros((num_inputs, n + 1))

    for i in range(num_inputs):
        truth_table[i, :n] = np.array(list(map(int, np.binary_repr(i, width=n)))) * 2 - 1

    for i in range(num_inputs):
        binary = np.array(list(map(int, np.binary_repr(i, width=n))))
        if np.any(binary):
            character_table = np.prod(truth_table[:, [idx for idx in range(n) if binary[idx] == 1]], axis=1)
            truth_table[:, n] += coeffs[i] * character_table

    truth_table[:, n] = np.sign(truth_table[:, n])
    truth_table[truth_table[:, n] == 0, n] = 1
    return truth_table

def sum_abs_values_of_terms_by_degree(coeffs):
    n = int(np.log2(len(coeffs)))
    degree_sums = np.zeros(n + 1)
    
    for i in range(len(coeffs)):
        binary = np.array(list(map(int, np.binary_repr(i, width=n))))
        degree = np.sum(binary)
        degree_sums[degree] += abs(coeffs[i])
    
    return degree_sums

n = 15
k=2
w=3

f_table = random_perm_read_k_BP(n,k,w)
f_vals = f_table[:, n].astype(int)
coeffs, fourier_expansion = boolean_fourier_expansion(f_vals)
truth_table = fourier_coeffs_to_boolean_function(coeffs)
print(fourier_expansion)

degree_sums = sum_abs_values_of_terms_by_degree(coeffs)
for d in range(1, n + 1):
    print(f"Sum of absolute values of terms with degree {d}: {degree_sums[d]}")
