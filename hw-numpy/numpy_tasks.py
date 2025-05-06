import numpy as np

def uniform_intervals(a, b, n):
    return np.linspace(a, b, n)
    pass

def cyclic123_array(n):
    return np.tile(np.array([1, 2, 3]), n)
    pass

def first_n_odd_number(n):
    return np.arange(1, 2 * n, 2, dtype=int)
    pass

def zeros_array_with_border(n):
    array = np.zeros((n,n))
    array = np.zeros((n, n), dtype = int)
    array[0, :] = 1
    array[-1, :] = 1
    array[:, 0] = 1
    array[:, -1] = 1
    return array
    pass

def chess_board(n):
    array = np.zeros((n,n))
    array[::2, ::2] = 1
    array[1::2, 1::2] = 1
    return array
    pass

def matrix_with_sum_index(n):
    array = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            array[i,j] = i+j
    return array
    pass

def cos_sin_as_two_rows(a, b, dx):
    x = np.arange(a, b, dx)
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.array([cos_x, sin_x])
    pass

def compute_mean_rowssum_columnssum(A):
    mean_value = np.mean(A)
    row_sums = np.sum(A, axis=0)
    column_sums = np.sum(A, axis=1)
    return mean_value, row_sums, column_sums
    pass

def sort_array_by_column(A, j):
    return A[np.argsort(A[:, j])]
    pass

def compute_integral(a, b, f, dx, method):
    x = np.arange(a, b + dx/2, dx)
    y = f(x)
    n = len(x)

    if method == 'rectangular':
        integral = np.sum(y[:-1] * dx)
    elif method == 'trapezoidal':
        integral = (dx / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    elif method == 'simpson':
        if n % 2 == 0:
             x = np.arange(a, b, dx)
             y = f(x)
             n = len(x)
             if n % 2 == 0:
                 x = np.arange(a, b - dx/2, dx)
                 y = f(x)
                 n = len(x)
             if n < 3:
                 raise ValueError("Для метода Симпсона необходимо минимум 3 точки.")
        integral = (dx / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

    else:
        raise ValueError("Неизвестный метод интегрирования.")

    return integral



    pass