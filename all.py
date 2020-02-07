import numpy as np
import time

# vanilla conjugate gradient
# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter

def cg(A, b, x, tol=1e-8, max_iter=10000):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p = 0, r

  while np.linalg.norm(err) > tol:

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

# conjugate gradient squared
def cgs(A, b, x, tol=1e-8, max_iter=10000):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p, u = 0, r, r
  r0 = r

  while np.linalg.norm(err) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(r0.T, Ap)
    alpha = err / pAp

    q = u - alpha * Ap
    x = x + alpha * (u + q)
    r = r - alpha * np.dot(A, (u + q))

    new_err = np.dot(r0.T, r)
    beta = new_err / err
    u = r + beta * q
    p = u + beta * (q + beta * p) # direction
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

# Biconjugate Gradient Stabilized (BICGSTAB)

def bicgstab(A, b, x, tol=1e-8, max_iter=10000):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p  = 0, r
  r0 = r

  while np.linalg.norm(r) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(r0.T, Ap)
    alpha = err / pAp

    s = r - alpha * Ap
    As = np.dot(A, s)
    omega = np.dot(s.T, As) / (np.dot(As.T, As))
    x = x + alpha * p + omega * s
    r = s - omega * As

    new_err = np.dot(r0.T, r)
    beta = (alpha / omega) * (new_err / err)
    p = r + beta * (p - omega * Ap)
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

# biconjugate gradient
# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter

def bicg(A, b, x, tol=1e-8, max_iter=10000):

  r = b - np.dot(A, x)   # residual
  q = b - np.dot(A.T, x) # residual

  err = np.dot(q.T, r) # error
  i, p, s, y = 0, r, q, x

  while np.linalg.norm(err) > tol:

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

if __name__ == "__main__":

  A = np.random.rand(50, 50)
  b = np.random.rand(50, 1)
  x0 = np.zeros((A.shape[1], 1))

  # apply A.T, make positive semidefinite
  t0 = time.perf_counter()
  A0 = np.dot(A.T, A)
  b0 = np.dot(A.T, b)

  print('CG')
  t0 = time.perf_counter()
  x, r, i = cg(A0, b0, x0)
  t1 = time.perf_counter()

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}\nerr norm={}'.format(max_err, err_norm))
  print('iters={}, time={:.6f} s'.format(i, t1-t0))
  x_cg = x

  print('BiCG')
  t0 = time.perf_counter()
  x, r, i = bicg(A0, b0, x0)
  t1 = time.perf_counter()

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}\nerr norm={}'.format(max_err, err_norm))
  print('iters={}, time={:.6f} s'.format(i, t1-t0))
  x_bicg = x

  print('CGS')
  t0 = time.perf_counter()
  x, r, i = cgs(A0, b0, x0)
  t1 = time.perf_counter()

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}\nerr norm={}'.format(max_err, err_norm))
  print('iters={}, time={:.6f} s'.format(i, t1-t0))
  x_cgs = x

  print('BICGSTAB')
  t0 = time.perf_counter()
  x, r, i = bicgstab(A0, b0, x0)
  t1 = time.perf_counter()

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}\nerr norm={}'.format(max_err, err_norm))
  print('iters={}, time={:.6f} s'.format(i, t1-t0))
  x_bicgstab = x
