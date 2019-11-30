if isinstance(other, ImageSet):
    #This is just a rough implementation.
    # Will be improved after discussing with mentor
    if len(self.lamda.variables) > 1 or len(other.lamda.variables) > 1:
        #Only simple lambda supported
        return None
    self_sym, other_sym = self.lamda.variables, other.lamda.variables
    self_expr, other_expr = self.lamda.expr, other.lamda.expr
    linear = Poly(self_expr, self_sym).is_linear and \
        Poly(other_expr, other_sym).is_linera
    if not linear:
        #Does not fulfill the needs of linearity
        return None
    base = other.base._set
    if base != self.base_set:
        # Base should be same
        return None
    result = unify_img(base, self_expr, other_expr, self_sym, other_sym)
        return result
