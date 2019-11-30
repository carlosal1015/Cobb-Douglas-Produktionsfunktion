# In solveset
if X:
    1
elif f.has(Mod) and not f.has(TrigonometricFunction):
    modterm = list(f.atoms(Mod))[0]
    t = Dummy('t', integer = True)
    f = f.xreplace({modterm: t})
    g_ys = solver(f, t, S.Integers)
    if isinstance(g_ys, FiniteSet):
        for rhs in g_ys:
            if not rhs.has(symbol) and modterm.has(symbol):
                result += _solve_modular(modterm, rhs, symbol, domain)

# Function _solve_modular

def _solve_modular(modterm, rhs, symbol, domain):
    "Helper function to solve modular equations"
    # Documentation to be added and more implementation also to be added
    #It gives idea of implementation
    a, m = modterm.args
    n = Dummy('n', real = True)

    if a is symbol:
        return ImageSet(Lambda(n, m*n + rhs%m), S.Integers)
    if a.is_Add:
        g, h = a.as_independent(symbol)
        u = Dummy('u', integer = True)
        sol_set = _solveset(u - rhs + Mod(g, m), u, domain)
        result = EmptySet()
        if isinstance(sol_set, FiniteSet):
            for s in sol_set:
                result += _solveset(Mod(h, m) - s, symbol, domain)
        elif isinstance(sol_set, ImageSet):
            soln = _solveset(Mod(h, m) - sol_set.lamda.expr, symbol, S.Integers)
        else:
            result = ConditionSet(symbol, Eq(modterm - rhs, 0), domain)
        return result
    if a.is_Mul:
        g, h = a.as_independent(symbol)
        return _solveset(Mod(h, m) - rhs*invert(g, m), symbol, domain)
    if a.is_Pow:
        if a.exp is symbol:
            return _solveset(a.exp - discrete_log(m, rhs, a.base),
                             symbol, domain)
    return ConditionSet(symbol, Eq(modterm - rhs, 0), domain)
