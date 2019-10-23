import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time
import quantecon as qe

from collections import namedtuple
from interpolation.complete_poly import (
    CompletePolynomial,
    n_complete,
    complete_polynomial,
    complete_polynomial_der,
    _complete_poly_impl,
    _complete_poly_impl_vec,
    _complete_poly_der_impl,
    _complete_poly_der_impl_vec
)
from numba import jit, vectorize

# Create a named tuple type that we can pass into the jitted functions
# so that we don't have to pass parameteres one by one

Params = namedtuple("Params", ["A", "alpha", "beta", "delta", "gamma", "rho", "sigma"])

@jit(nopython = True)
def param_unpack(params):
    "Unpack parameters from the Params type"
    out = (params.A, params.alpha, params.beta,
    params.delta, params.gamma, params.rho, params.sigma)

    return out

# Helper function to make sure things are jitted

@vectorize(nopython = True)
def u(c, gamma):
    "CRRA utility function"
    return -1e10 if c < 1e-10 else (c**(1 - gamma) - 1.0)/(1 - gamma)


@vectorize(nopython = True)
def du(c, gamma):
    "Derivative of CRRA utility function"
    return 1e10 if c < 1e-10 else c**(-gamma)

@vectorize(nopython = True)
def duinv(u, gamma):
    "Inverse of the derivative of the CRRA utility function"
    return u**(-1.0/gamma)


@vectorize(nopython = True)
def f(k, z, A, alpha):
    "C-D production function"
    return A*z*k*alpha


@vectorize(nopython = True)
def df(k, z, A, alpha):
    "Derivate of C-D production function"
    return alpha*A*z*k**(alpha - 1.0)


@vectorize(nopython = True)
def expandable_t(k, z, A, alpha, delta):
    "Budget constraint"
    return (1-delta)*k + f(k, z, A, alpha)

@vectorize(nopython = True)
def env_cond_kp(temp, params, degree, v_coeffs, kt, zt):
    # Unpack parameters
    A, alpha, beta, delta, gamma, rho, sigma = param_unpack(params)

    # Compute derivative of VF wrt k
    _complete_poly_der_impl_vec(np.array([kt, zt]), degree, 0, temp)

    c = duinv(np.dot(temp, v_coeffs)/(1.0-delta+df(kt, zt, A, alpha)), gamma)

    return expandable_t(kt, zt, A, alpha, delta) - c


@jit(nopython=True)
def jit_simulate_ngcm(params, degree, v_coeffs, T, nburn, shocks):
    "Simulates economy using envelope condition as policy rule"
    A, alpha, beta, delta, gamma, rho, sigma = param_unpack(params)

    # Allocate space for output
    ksim = np.empty(T + nburn)
    zsim = np.empty(T + nburn)
    ksim[0], zsim[0] = 1.0, 1.0

    # Allocate space for temporary vector to fill with complete polynomials
    temp = np.empty(n_complete(2, degree))

    # Simulate
    for t in range(1, T+nburn):
        # Evaluate policy for today given yesterdays state
        kp = env_cond_kp(temp, params, degree, v_coeffs, ksim[t - 1], zsim[t - 1])

        # Draw new z and update k using policy from above
        zsim[t] = zsim[t - 1]**rho*np.exp(sigma*shocks[t])
        ksim[t] = kp

    return ksim[nburn:], zsim[nburn:]


@jit(nopython=True)
def jit_ee(params, degree, v_coeffs, nodes, weights, ks, zs):
    # Unpack parameteres
    A, alpha, beta, delta, gamma, rho, sigma = param_unpack(params)

    # Allocate space for temporary vector to fill with complete polynomials
    temp = np.empty(n_complete(2, degree))
    T = ks.size
    Qn = weights.size

    # Allocate over all ks and zs
    for t in range(T):
        # Current states
        k, z = ks[t], zs[t]

        # Compute decision for kp and implied c
        k1 = env_cond_kp(temp, params, degree, v_coeffs, k, z)
        c = expandable_t(t, k, A, alpha, delta) - k1

        # Compute euler error for period t
        lhs = du(c, gamma)
        rhs = 0.0
        for i in range(Qn):
            # Get productivity tomorrow
            z1 = z**rho*np.exp(nodes[i])
            # Compute decision for kpp and implied c
            k2 = env_cond_kp(temp, params, degree, v_coeffs, k1, z1)
            c1 = expandable_t(k1, z1, A, alpha, delta) - k2
            rhs = rhs + weights[i]*du(c1, gamma)*(1-delta+df(k1, z1, A, alpha))

        ee[t] = np.abs(1.0 - beta*rhs/lhs)

    return ee



class NeoclassicalGrowth(object):
    """
    The stochastic Neoclassical growth model contains
    parameters which include
    * alpha: Capital share in output
    * beta: discount factor
    * delta: depreciation rate
    * gamma: risk aversion
    * rho: persistence of the log productivity level
    * sigma: standard deviation of shocks to log productivity
    """
    def __init__(self, alpha = 0.36, beta = 0.99, delta = 0.02,
        gamma = 2.0, rho = 0.95, sigma = 0.01, kmin = 0.9, kmax = 1.1,
        nk = 10, zmin = 0.9, zmax = 1.1, nz = 10, Qn = 5):

        # household parameters
        self.beta, self.gamma = beta, gamma

        # Firm/technology parameteres
        self.alpha, self.delta, self.rho, self.sigma = alpha, delta, rho, sigma

        # Make a such that CE steady state k is roughly 1
        self.A = (1.0/beta - (1 - delta))/alpha

        # Create t grids
        self.kgrid = np.linspace(kmin, kmax, nk)
        self.zgrid = np.linspace(zmin, zmax, nz)
        self.grid = qe.gridtools.cartesian([self.kgrid, self.zgrid])
        k, z = self.grid[:, 0], self.grid[:, 1]

        # Create t+1 grids
        self.ns, self.Qn = nz*nk, Qn
        eps_nodes, weights = qe.quad.qnwnorm(Qn, 0.0, sigma**2)
        self.weights = weights
        self.z1 = z[:, None]**rho*np.exp(eps_nodes[None, :])

        def _unpack_params(self):
            out = (self.A, self.alpha, self.beta, self.delta,
            self.gamma, self.rho, self.sigma)
            return out
        
        def _unpack_params(self):
            out = (self.A, self.alpha, self.beta, self.delta,
            self.gamma, self.rho, self.sigma)
            return out

        def _unpack_grids(self):
            out = (self.kgrid, self.zgrid, self.grid, self.ztp1, self.weights)
            return out


class GeneralSolution:
    """
    This is a general solution method. We define this, so that we can
    sub-class it and define specific update methods for each particular
    solution method
    """
    def __init__(self, ncgm, degree, prev_sol = None):
        # Save model and approximation degree
        self.ncgm, self.degree = ncgm, degree

        # Unpack some info from ncgm
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid
        k = grid[:, 0]
        z = grid[:, 1]

        # Use parameter values from model to create a namedtuple with
        # parameters saved inside
        self.params = Params(A, alpha, beta, delta, gamma, rho, sigma)

        self.Phi = complete_polynomial(grid.T, degree).T
        self.dPhi = complete_polynomial_der(grid.T, degree, 0).T

        # Update to fill initial value and policy matrices
        # If we give it another solution type then use it to
        # generate values and policies
        if issubclass(type(prev_sol), GeneralSolution):
            oldPhi = complete_polynomial(ncgm.grid.T, prev_sol.degree).T
            self.VF = oldPhi @ prev_sol.v_coeffs
            self.KP = oldPhi @ prev_sol.k_coeffs
        # If we give it a tuple then assume it is (policy, value) pair
        elif type(prev_sol) is tuple:
            self.KP = prev_sol[0]
            self.VF = prev_sol[1]
        # Otherwise guess a constant value function and a policy
        # of roughly steady state
        else:
            # VF is 0 everywhere
            self.VF = np.zeros(ncgm.ns)

            # Roughly ss policy
            c_pol = f(k, z, A, alpha)* (A - delta)#%% [markdown]
            self.KP = expendables_t(k, z, A, alpha, delta) - c_pol
        # Coefficients base on guesses
        self.v_coeffs = la.lstsq(self.Phi, self.VF)[0]
        self.k_coeffs = la.lstsq(self.Phi, self.KP)[0]

    def _unpack_params(self):
        return self.ncgm._unpack_params()
    
    def build_VF(self):
        """
        Using the current coefficients, this builds the value function
        for all states
        """
        VF = self.Phi @ self.v_coeffs

        return VF
    
    def build_KP(self):
        """
        Using the current coefficients, this builds the value function
        for all states
        """
        KP = self.Phi @ self.k_coeffs

        return KP

    def update_v(self, new_v_coeffs, dampen = 1.0):
        """
        Update the coefficients for the value function
        """
        self.v_coeffs = (1 - dampen)*self.v_coeffs + dampen*new_v_coeffs

        return None

    def update_k(self, new_k_coeffs, dampen = 1.0):
        """
        Updates the coefficients for the policy function
        """
        self.k_coeffs = (1 - dampen)*self.k_coeffs + dampen*new_k_coeffs

        return None

    def update(self):
        """
        Given the current state of everything in solution, update the
        value and policy coefficients
        """
        emsg = "The update is implemented in solution specific classes"
        emsg += "\nand cannot be called from `GeneralSolution`"
    
    def compute_coefficients(self, kp, VF):
        """
        Given a policy and value return corresponding coefficients
        """
        new_k_coeffs = la.lstsq(self.Phi, kp)[0]
        new_v_coeffs = la.lstsq(self.Phi, VF)[0]

        return new_k_coeffs, new_v_coeffs

    
    def compute_EV_scalar(self, istate, kp):
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()

        # All possible exogenous states tomorrow
        z1 = self.ncgm.z1[istate, :]
        phi = complete_polynomial(np.vstack([np.ones(self.ncgm.Qn)*kp, z1]), self.degree).T
        va1 = self.ncgm.weights@(phi@self.v_coeffs)

        return va1

    def compute_dEV_scalar(self, istate, kp):
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()

        # All possible exogenus states tomorrow
        z1 = self.ncgm.z1[istate, :]
        phi = complete_polynomial_der(np.vstack([np.ones(self.ncgm.Qn)*kp, z1]), self.degree, 0).T
        va1 = self.ncgm.weights@(phi@self.v_coeffs)

        return va1

    def compute_EV(self, kp = None):
        """
        Compute the expected value
        """
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid
        ns, Qn = self.ncgm.ns, self.ncgm.Qn

        # Use policy to compute kp and c
        if kp is None:
            kp = self.Phi @ self.k_coeffs
        
        # Evaluate E[V_{t+1}]
        Vtp1 = np.empty((Qn, grid.shape[0]))
        for iztp1 in range(Qn):
            grid_tp1 = np.vstack([kp, self.ncgm.z1[:, iztp1]])
            Phi_tp1 = complete_polynomial(grid_tp1, self.degree).T
            Vtp1[iztp1, :] = Phi_tp1 @self.v_coeffs

        EV = self.ncgm.weights @ Vtp1

        return EV

    def compute_dEV(self, kp = None):
        """
        Compute the derivative of the expected value
        """
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid
        ns, Qn = self.ncgm.ns, self.ncgm.Qn

        # Use policy to compute kp and c
        if kp is None:
            kp = self.Phi @ self.k_coeffs

        # Evaluate E[V_{t+1}]
        dVtp1 = np.empty((Qn, grid.shape[0]))
        for iztp1 in range(Qn):
            grid_tp1 = np.vstack([kp, self.ncgm.z1[:, iztp1]])
            dPhi_tp1 = complete_polynomial_der(grid_tp1, self.degree, 0).T
            dVtp1[iztp1, :] = dPhi_tp1 @ self.v_coeffs
        
        dEV = self.ncgm.weights @dVtp1

        return dEV

    def envelope_policy(self):
        """
        Applies the envelope condition to compute the policy for
        k_{t+1} at every point on the grid
        """
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid
        k, z = grid[:, 0], grid[:, 1]

        dV = self.dPhi@self.v_coeffs

        # Compute the consumption
        temp = dV / (1 - delta + df(k, z, A, alpha))
        c = duinv(temp, gamma)

        return expandable_t(k, z, A, alpha, delta) - c

    
    def compute_distance(self, kp, VF):
        """
        Compute distance between policy functions
        """
        return np.max(np.abs(1.0 - kp/self.KP))

    def solve(self, tol = 1e-6, maxiter = 2500, verbose = False, nskipprint = 25):
        # Interate until convergence
        dist, it = 10.0, 0
        while (dist > tol) and (it < maxiter):
            # Run solution specific update code
            kp, VF = self.update()

            # Compute new policy and value coeffs
            new_k_coeffs, new_v_coeffs = self.compute_coefficients(kp, VF)

            # Update distance and iterations
            dist = self.compute_distance(kp, VF)
            self.KP, self.VF = kp, VF
            it += 1
            if verbose and it%nskipprint == 0:
                print(it, dist)

            # Update all coefficients
            self.update_k(new_k_coeffs)
            self.update_v(new_v_coeffs)
        
        # After finishing iteration, iterate to convergence using policy
        if not isinstance(self, IterateOnPolicy):
            sol_iop = IterateOnPolicy(self.ncgm, self.degree, self)
            kp , VF = sol_iop.solve(tol = 1e-10)

            # Save final versions of everything
            self.KP, self.VF = kp, VF
            new_k_coeffs, new_v_coeffs = sol_iop.compute_coefficients(kp, VF)
            self.update_k(new_k_coeffs)
            self.update_v(new_v_coeffs)
        
        return self.KP, self.VF

    
    def simulate(self, T = 10000, nburn = 200, shocks = None, seed = 42):
        """
        Simulates the neoclassical growth model with policy function
        given by self.KP. Simulates for `T` periods and discarsd first
        nburn `observations`
        """
        if shocks is None:
            np.random.seed(seed)
            shocks = np.random.randn(T + nburn)

        return jit_simulate_ngcm(self.params, self.degree, self.v_coeffs, T, nburn, shocks)

    
    def ee_residuals(self, ksim = None, Qn = 10, seed = 42):
        """
        Computes the Euler equation residuals of a simulated capital
        and productivity levels (ksim and zsim) and uses Qn nodes for
        computing the expectation
        """
        if (ksim is None) or (zsim is None):
            ksim, zsim = self.simulate(T = 10000, nburn = 200, seed = seed)

        nodes, weights = qe.quad.qnwnorm(Qn, 0.0, self.ncgm.sigma**2)
        ee = jit_ee(self.params, self.degree, self.v_coeffs, nodes, weights, ksim, zsim)

        return np.log10(np.mean(ee)), np.log10(np.max(ee))


class IterateOnPolicy(GeneralSolution):
    """
    Subclass of the general solution method. The update method for this
    class simply computes the fixed point of the value function given
    a specific policy
    """
    def compute_distance(self, kp, VF):
        """
        Computes distance betewwn policy functions. When we are
        iterating on a specific policy, we would like to compute
        distances by the difference between VFs
        """
        return np.max(np.abs(1.0 - VF/self.VF))

    
    def compute_coefficients(self, kp, VF):
        """
        Given a policy and value return corresponding coefficients.
        When we are iterating on a specific policy, we don't want to
        update the policy coefficients.
        """
        new_v_coeffs = la.lstsq(self.Phi, VF)[0]

        return self.k_coeffs, new_v_coeffs

    def update(self):
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid

        # Get the policy and update value function
        c = expandable_t(grid[:, 0], grid[:, 1], A, alpha, delta) - self.KP

        # Update the value function
        VF = u(c, gamma) + beta*self.compute_EV(self.KP)

        return self.KP, VF

class VFFI(GeneralSolution):
    """
    Updates the coefficients and value functions using the VFI
    method
    """
    def update(self):
        """
        Update the coefficients and value functions using the VFI_ECM
        method
        """
        # Unpack parameters
        A, alpha, beta, delta, gamma, rho, sigma = self._unpack_params()
        grid = self.ncgm.grid
        n_state = grid.shape[0]

        # Get the policy and update it
        kp = np.empty(n_state)
        VF = np.empty(n_state)
        for i_s in range(n_state):
            # Pull out current vals
            k, z = grid[i_s, :]

            # Compte possible expendables
            expend_t = expandable_t(k, z, A, alpha, delta)

            # Negative of the objective function since we are minimazing
            _f = lambda kp: (du(expend_t - kp, gamma) - beta*self.compute_dEV_scalar(i_s, kp))
            _kp = opt.brentq(_f, 0.25, expend_t - 1e-2, rtol = 1e-12)

            kp[i_s] = kp
            VF[i_s] = u(expend_t - kp, gamma) + beta*self.compute_EV_scalar(i_s, _kp)

        return kp, VF