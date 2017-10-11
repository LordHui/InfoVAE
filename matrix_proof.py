import numpy as np


T_lambda = np.array([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  -1, 0],
    [0,  -1, 0,  0,  0],
    [0,  0,  0,  0,  -1],
    [0,  0,  -1,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  1,  0],
    [1,  1,  0,  0,  0],
    [0,  0,  0,  0,  1],
    [-1, 0,  1,  0,  0]
], dtype=np.float)

T_lb = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  0],
    [0,  0,  0,  0],
    [0,  0,  0,  0],
    [0,  0,  1,  0],
    [0,  0,  0,  0],
    [0,  0,  0,  0],
    [0,  0,  0,  1],
    [0,  0,  0,  0],
    [0,  0,  0,  0],
], dtype=np.float)

T_ulf = np.concatenate([T_lb, np.array([
    [0,  0],
    [0,  0],
    [0,  0],
    [-1, 0],
    [0,  -1],
    [0,  0],
    [0,  0],
    [0,  0],
    [1,  0],
    [0,  1],
], dtype=np.float)], axis=1)

T_blf = np.concatenate([T_ulf, np.array([

], dtype=np.float)], axis=1)

T_q = np.array([
    [1,  1,  0,  0,  0],
    [1,  0,  0,  0,  0],
    [0,  1,  0,  0,  0],
    [0,  1,  0,  0,  0],
    [1,  0,  1,  0,  0],
    [0,  0,  0,  1,  1],
    [0,  0,  0,  1,  0],
    [0,  0,  0,  0,  1],
    [0,  0,  0,  0,  1],
    [0,  0,  0,  1,  0],
], dtype=np.float)


solution_lb = np.dot(np.linalg.inv(np.dot(T_lambda.transpose(), T_lambda)), T_lambda.transpose())
print(solution_lb)
