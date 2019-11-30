if mainpow and symbol in mainpow.exp.free_symbols:
    lhs = collect(lhs, mainpow)
    if lhs.is_Mul and rhs != 0:
        soln = _lambert(expand_log(log(abs(lhs)) - log(abs(rhs))), symbol)
# see the abs used.
