
from sympy import Symbol, acos, sqrt, atan
D = Symbol('D', positive = True)
e = acos(D) + 2*atan(D/(sqrt(-D + 1)*sqrt(D + 1))
# -1/sqrt(-D**2 + 1)
trigsimp(e)
# acos(D) + 2*atan(D/sqrt(-D + 1)*sqrt(D + 1))
simplify(e)
# acos(D) + 2*atan(D/sqrt(-D + 1)*sqrt(D + 1)) -1/sqrt(-D**2 + 1))
