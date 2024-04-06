import pandas as pd
import numpy as np


def partial(func: "Function", vals: tuple, index, delta = 0.00001):
    vals = np.array(vals, dtype = "float")
    vals_delta = vals.copy()
    vals_delta[index] += delta
    
    return (func(*vals_delta) - func(*vals)) / delta
#%%
def grad(func: "Function", vals: tuple, delta = 0.0000001):
    return np.array([partial(func, vals, i, delta = delta) for i in range(len(vals))])

def gradient_des(func: "Function", x0: tuple,
                 epsilon: float = 0.001,
                 max_iter: int = 10_000,
                 alpha: float = 0.3,
                 delta: float = 1e-7,
                 verbose: bool = False, 
                 orbit: bool = False) -> np.ndarray:
    x0 = np.array(x0)
    x1 = x0 - alpha * grad(func, x0, delta = delta)
    
    if verbose:
        print(x0)
    
    points = [x0, x1]
    
    
    i = 1
    
    while np.linalg.norm(x1 - x0) > epsilon and i <= max_iter:
        x0 = x1
        x1 = x0 - alpha * grad(func, x0, delta = delta)
        
        points.append(x1)
        

        i += 1
        
    if verbose:
        print('Max Iter: {}'.format(i))
        print('Norm: {}'.format(np.linalg.norm(x1 - x0)))
    
    if orbit:
        return x1, points
        
    return x1