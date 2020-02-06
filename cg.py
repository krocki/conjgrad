import numpy as np
import time

# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter
def conjugate_gradients(A, b, x, tol=1e-6, max_iter=10000):

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

def solve(A, b):

  x0 = np.zeros(A.shape[1])
  # apply A.T, make positive semidefinite
  t0 = time.perf_counter()
  A0 = np.dot(A.T, A)
  b0 = np.dot(A.T, b)
  t1 = time.perf_counter()
  x, r, i = conjugate_gradients(A0, b0, x0)
  t2 = time.perf_counter()

  t = {"prepro" : t1-t0, "cg" : t2-t1}

  return x, r, i, t

if __name__ == "__main__":

  A = np.random.rand(40, 50)
  b = np.random.rand(40)
  x,r,i,t = solve(A, b)

  # verify
  Ax = np.dot(A, x)
  max_err = np.max(np.fabs(b - Ax))
  err_norm = np.linalg.norm(b - Ax)
  print('max err={}, err norm]=={}'.format(max_err, err_norm))
  print('i={}, t={:.6f} s'.format(i, t['cg']))

# use OCTAVE as ref
  #A = np.loadtxt("A.txt")
  #b = np.loadtxt("b.txt")
  #x_ref = np.loadtxt("x.txt")
  #t_ref = np.loadtxt("t.txt")

  ## solution, residuals, iters, time
  #x,r,i,t = solve(A, b)

  #print('r={}'.format(r))

  #max_err = np.max(np.fabs(x - x_ref))
  #err_norm = np.linalg.norm(x - x_ref)

  #print('max err={}, err norm]=={}'.format(max_err, err_norm))
  #print('i={}, t={:.6f} s, ref t={:.6f} s'.format(i,t['cg'], t_ref))
