import numpy as np
import time

# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter
def conjugate_gradients(A, b, x, tol=1e-8, max_iter=0):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p = 0, r

  while np.sqrt(err) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(p.T, Ap)
    alpha = err / pAp

    x = x + alpha * p
    r = r - alpha * Ap

    new_err = np.dot(r.T, r)
    beta = new_err / err
    p = r + beta * p # direction
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

if __name__ == "__main__":

  #A = np.array([[4, 1],[1, 3]])
  A = np.random.rand(10, 20)
  #b = np.array([[1],[2]])
  b = np.random.rand(10)

  # initial solution
  x0 = np.zeros(A.shape[1])
  # apply A.T, make positive semidefinite
  A0 = np.dot(A.T, A)
  b0 = np.dot(A.T, b)

  t0 = time.perf_counter()
  x, r, i = conjugate_gradients(A0, b0, x0)
  t1 = time.perf_counter()

  print('x')
  print(x)
  print('i={}, r={}, t={:.6f} s'.format(i, r, t1-t0))
