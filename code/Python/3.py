from __future__ import division
%pylab --no-import-all
%matplotlib inline
from scipy import interp

interp?

x=np.linspace(0, np.pi, 100)
plt.figure(1)
plt.plot(x, np.sin(x), label = 'Actual Function')
for  i in np.arange(3, 11, 2):
	plt.figure(1)
	xp=np.linspace(0, np.pi, i)
	yp=np.sin(xp)
	y=interp(x, xp, yp)
	plt.plot(x, y, label = 'Interpolation ' + str(i))
	plt.figure(2)
	plt.title('Error with up to ' + str(i) + ' points in interpolation')
	plt.ylabel('Error')
	plt.plot(y - np.sin(x), label = str(i))
	plt.legend(loc = 8)
plt.figure(1)
plt.legend(loc = 8)
plt.show()


class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.
        """
        self.X, self.Y = X, Y

    def __call__(self, z):
        """Parameters: z is a number, sequence or array.
        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.
        """
        if isinstance(z, int) or isinstance(z, float):
            return interp ([z], self.X, self.Y)[0]
        else:
            return interp(z, self.X, self.Y)


xp = np.linspace(0, np.pi, 10)
yp = np.sin(xp)
oursin = LinInterp(xp, yp)
plt.plot(oursin(x))

%%file optgrowthfuncs.py
def U(c,sigma=1):
    '''This function returns the value of utility when the CRRA
    coefficient is sigma. I.e. 
    u(c,sigma)=(c**(1-sigma)-1)/(1-sigma) if sigma!=1 
    and 
    u(c,sigma)=ln(c) if sigma==1
    Usage: u(c,sigma)
    '''
    if sigma!=1:
        u=(c**(1-sigma)-1)/(1-sigma)
    else:
        u=np.log(c)
    return u

def F(K,L=1,alpha=.3,A=1):
    '''
    Cobb-Douglas production function
    F(K,L)=K^alpha L^(1-alpha)
    '''
    return K**alpha * L**(1-alpha)

def Va(k,alpha=.3,beta=.9):
    ab=alpha*beta
    return np.log(1-ab)/(1-beta)+ab*np.log(ab)/((1-beta)*(1-ab))+alpha*np.log(k)/(1-ab)

def opk(k,alpha=.3,beta=.9):
    return alpha*beta*k**alpha

def opc(k,alpha=.3,beta=.9):
    return (1-alpha*beta)*k**alpha

%load optgrowthfuncs.py

def U(c,sigma=1):
    '''This function returns the value of utility when the CRRA
    coefficient is sigma. I.e. 
    u(c,sigma)=(c**(1-sigma)-1)/(1-sigma) if sigma!=1 
    and 
    u(c,sigma)=ln(c) if sigma==1
    Usage: u(c,sigma)
    '''
    if sigma!=1:
        u=(c**(1-sigma)-1)/(1-sigma)
    else:
        u=np.log(c)
    return u

def F(K,L=1,alpha=.3,A=1):
    '''
    Cobb-Douglas production function
    F(K,L)=K^alpha L^(1-alpha)
    '''
    return K**alpha * L**(1-alpha)

def Va(k,alpha=.3,beta=.9):
    ab=alpha*beta
    return np.log(1-ab)/(1-beta)+ab*np.log(ab)/((1-beta)*(1-ab))+alpha*np.log(k)/(1-ab)

def opk(k,alpha=.3,beta=.9):
    return alpha*beta*k**alpha

def opc(k,alpha=.3,beta=.9):
    return (1-alpha*beta)*k**alpha

alpha=.3
beta=.9
sigma=1
delta=1

# Grid of values for state variable over which function will be approximated
gridmin, gridmax, gridsize = 0.1, 5, 300
grid = np.linspace(gridmin, gridmax**1e-1, gridsize)**10


plt.hist(grid,bins=50);
plt.xlabel('State Space');
plt.ylabel('Number of Points');

plt.plot(grid,grid,'r.');
plt.title('State Space Grid');


from scipy.optimize import fminbound
fminbound?

# Maximize function V on interval [a,b]
def maximum(V, a, b):
    return float(V(fminbound(lambda x: -V(x), a, b)))
# Return Maximizer of function V on interval [a,b]
def maximizer(V, a, b):
    return float(fminbound(lambda x: -V(x), a, b))

# The following two functions are used to find the optimal policy and value functions using value function iteration
# Bellman Operator
def bellman(w):
    """The approximate Bellman operator.
    Parameters: w is a LinInterp object (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
    w is a function defined on the state space.
    """
    vals = []
    for k in grid:
        kmax=F(k,alpha=alpha)
        h = lambda kp: U(kmax + (1-delta) * k - kp,sigma) + beta * w(kp)
        vals.append(maximum(h, 0, kmax))
    return LinInterp(grid, vals)

# Optimal policy
def policy(w):
    """
    For each function w, policy(w) returns the function that maximizes the 
    RHS of the Bellman operator.
    Replace w for the Value function to get optimal policy.
    The approximate optimal policy operator w-greedy (See Stachurski (2009)). 
    Parameters: w is a LinInterp object (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that captures the optimal policy.
    """
    vals = []
    for k in grid:
        kmax=F(k,alpha=alpha)
        h = lambda kp: U(kmax + (1-delta) * k - kp,sigma) + beta * w(kp)
        vals.append(maximizer(h, 0, kmax))
    return LinInterp(grid, vals)

# Parameters for the optimization procedures
count = 0
maxiter = 1000
tol = 1e-6
print('tol = %f' % tol)

V0 = LinInterp(grid,U(grid))
plt.figure(1)
plt.plot(grid,V0(grid), label = 'V' + str(count));
plt.plot(grid,Va(grid), label = 'Actual');
plt.legend(loc = 6);

plt.plot(grid, V0(grid), label = 'V' + str(count))
count += 1
V0 = bellman(V0)
plt.figure(1)
plt.plot(grid, V0(grid), label = 'V' + str(count))
plt.plot(grid, Va(grid), label = 'Actual')
plt.legend(loc = 6);
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(grid.min(), grid.max())
ax.plot(grid,Va(grid), label='Actual', color='k', lw=2, alpha=0.6);

count = 0
maxiter = 200
tol = 1e-6
while count < maxiter:
    V1 = bellman(V0)
    err = np.max(np.abs(np.array(V1(grid)) - np.array(V0(grid))))
    if np.mod(count, 10) == 0:
        ax.plot(grid, V1(grid), color = plt.cm.jet(count / maxiter), lw = 2, alpha = 0.6)
        #print '%d %2.10f ' % (count,err)
    V0 = V1
    count += 1
    if err < tol:
        print(count)
        break
ax.plot(grid, V1(grid), label = 'Estimated', color = 'r', lw = 2, alpha = 0.6)
ax.legend(loc = 'lower right')
plt.draw()