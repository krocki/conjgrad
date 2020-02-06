function ref(M, N, fA, fB, fX, fT)

  A = rand(M, N);
  b = rand(M, 1);

  tic;
  x = bicgstab(A, b, rtol=1e-6, maxit=1000);
  t=toc;

  save ("-ascii", fA, "A");
  save ("-ascii", fB, "x");
  save ("-ascii", fX, "b");
  save ("-ascii", fT, "t");

end
