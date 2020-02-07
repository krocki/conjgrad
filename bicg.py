import numpy as np
import time

# vanilla conjugate gradient
# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter
def cg(A, b, x, tol=1e-6, max_iter=10000):

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

# biconjugate gradient
# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter

def bicg(A, b, x, tol=1e-6, max_iter=10000):

  r = b - np.dot(A, x)   # residual
  q = b - np.dot(A.T, x) # residual

  err = np.dot(q.T, r) # error
  i, p, s, y = 0, r, q, x

  while np.sqrt(np.fabs(err)) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(s.T, Ap)
    alpha = err / pAp

    x = x + alpha * p
    y = y + alpha * s

    r = r - alpha * Ap
    q = q - alpha * np.dot(A.T, s)

    new_err = np.dot(q.T, r)
    beta = new_err / err
    p = r + beta * p # direction
    s = q + beta * s
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

def solve(A, b, func):

  x0 = np.zeros((A.shape[1], 1))
  # apply A.T, make positive semidefinite
  t0 = time.perf_counter()
  A0 = np.dot(A.T, A)
  b0 = np.dot(A.T, b)
  t1 = time.perf_counter()
  x, r, i = func(A0, b0, x0)
  t2 = time.perf_counter()

  t = {"prepro" : t1-t0, "cg" : t2-t1}

  return x, r, i, t

if __name__ == "__main__":

  A = np.random.rand(40, 40)
  b = np.random.rand(40, 1)
  print('bicg')
  x,r,i,t = solve(A, b, bicg)

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}, err norm]=={}'.format(max_err, err_norm))
  print('i={}, t={:.6f} s'.format(i, t['cg']))

  x,r,i,t = solve(A, b, cg)

  # verify
  print('cg')
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}, err norm]=={}'.format(max_err, err_norm))
  print('i={}, t={:.6f} s'.format(i, t['cg']))
