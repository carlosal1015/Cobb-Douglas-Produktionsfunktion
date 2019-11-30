solveset(sin(2*x) + sin(4*x) + sin(6*x), x, S.Reals)
###########################################
Union(ImageSet(Lambda(_n, 2*_n*pi), Integers),
      ImageSet(Lambda(_n, 2*_n*pi + pi), Integers),
      ImageSet(Lambda(_n, 2*_n*pi + pi/2), Integers),
      ImageSet(Lambda(_n, 2*_n*pi + 3*pi/2), Integers),
      ImageSet(Lambda(_n, 2*_n*pi + 4*pi/2), Integers))

# this should be
Union(ImageSet(Lambda(_n, _n*pi/4), Integers),
      ImageSet(Lambda(_n, _n*pi + pi/6), Integers),
      ImageSet(Lambda(_n, _n*pi - pi/6), Integers))

solveset(sin(x), x)
######################################
Union(ImageSet(Lambda(_n, 2*_n*pi), Integers),
      ImageSet(Lambda(_n, 2*_n*pi + pi), Integers))

# this should be
ImageSet(Lambda(_n, _n*pi), Integers)
