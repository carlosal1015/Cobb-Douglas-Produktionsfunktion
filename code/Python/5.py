n = symbols('n', integer = True)
a = 742938285
z = 1898888478
m = 2**31 - 1
x = 20170816
solveset(x - Mod(a**n*z, m), n, S.Integers)
