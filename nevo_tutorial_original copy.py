#!/usr/bin/env python
# coding: utf-8

# # Random Coefficients Logit Tutorial with the Fake Cereal Data

# In[ ]:


import pyblp
import numpy as np
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False
pyblp.__version__


# In this tutorial, we'll use data from [Nevo (2000a)](https://pyblp.readthedocs.io/en/stable/references.html#nevo-2000a) to solve the paper's fake cereal problem. Locations of CSV files that contain the data are in the [`data`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.data.html#module-pyblp.data) module.
# 
# ## Theory of Random Coefficients Logit
# 
# The random coefficients model extends the plain logit model by allowing for correlated tastes for different product characteristics.
# In this  model (indirect) utility is given by
# 
# $$u_{ijt} = \alpha_i p_{jt} + x_{jt} \beta_i^\text{ex} + \xi_{jt} + \epsilon_{ijt}$$
# 
# The main addition is that $\beta_i = (\alpha_i, \beta_i^\text{ex})$ have individual specific subscripts $i$.
# 
# Conditional on $\beta_i$, the individual market share follow the same logit form as before. But now we must integrate over heterogeneous individuals to get the aggregate market share:
# 
# $$s_{jt}(\alpha, \beta, \theta) = \int \frac{\exp(\alpha_i p_{jt} + x_{jt} \beta_i^\text{ex} + \xi_{jt})}{1 + \sum_k \exp(\alpha_i p_{jt} + x_{kt} \beta_i^\text{ex} + \xi_{kt})} f(\alpha_i, \beta_i \mid \theta).$$
# 
# In general, this integral needs to be calculated numerically. This also means that we can't directly linearize the model. It is common to re-parametrize the model to separate the aspects of mean utility that all individuals agree on, $\delta_{jt} = \alpha p_{jt} + x_{jt} \beta^\text{ex} + \xi_{jt}$, from the individual specific heterogeneity, $\mu_{ijt}(\theta)$. This gives us
# 
# $$s_{jt}(\delta_{jt}, \theta) = \int \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_k \exp(\delta_{kt} + \mu_{ikt})} f(\mu_{it} | \theta).$$
# 
# Given a guess of $\theta$ we can solve the system of nonlinear equations for the vector $\delta$ which equates observed and predicted market share $s_{jt} = s_{jt}(\delta, \theta)$. Now we can perform a linear IV GMM regression of the form
# 
# $$\delta_{jt}(\theta) = \alpha p_{jt} + x_{jt} \beta^\text{ex} + \xi_{jt}.$$
# 
# The moments are constructed by interacting the predicted residuals $\hat{\xi}_{jt}(\theta)$ with instruments $z_{jt}$ to form
# 
# $$\bar{g}(\theta) =\frac{1}{N} \sum_{j,t} z_{jt}' \hat{\xi}_{jt}(\theta).$$

# ## Random Coefficients
# 
# To include random coefficients we need to add a [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation) for the demand-side nonlinear characteristics $X_2$.
# 
# Just like in the logit case we have the same reserved field names in `product_data`:
# 
# - `market_ids` are the unique market identifiers which we subscript $t$.
# - `shares` specifies the market share which need to be between zero and one, and within a market ID, $\sum_{j} s_{jt} < 1$.
# - `prices` are prices $p_{jt}$. These have some special properties and are _always_ treated as endogenous.
# - `demand_instruments0`, `demand_instruments1`, and so on are numbered demand instruments. These represent only the _excluded_ instruments. The exogenous regressors in $X_1$ (of which $X_2$ is typically a subset) will be automatically added to the set of instruments.
# 
# We proceed with the following steps:
# 
# 1. Load the `product data` which at a minimum consists of `market_ids`, `shares`, `prices`, and at least a single column of demand instruments, `demand_instruments0`.
# 2. Define a [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation) for the $X_1$ (linear) demand model.
# 
#     - This and all other formulas are similar to R and [patsy](https://patsy.readthedocs.io/en/stable/) formulas.
#     - It includes a constant by default. To exclude the constant, specify either a `0` or a `-1`.
#     - To efficiently include fixed effects, use the `absorb` option and specify which categorical variables you would like to absorb.
#     - Some model reduction may happen automatically. The constant will be excluded if you include fixed effects and some precautions are taken against collinearity. However, you will have to make sure that differently-named variables are not collinear.
# 
# 3. Define a [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation) for the $X_2$ (nonlinear) demand model.
# 
#     - Include only the variables over which we want random coefficients.
#     - Do not absorb or include fixed effects.
#     - It will include a random coefficient on the constant (to capture inside good vs. outside good preference) unless you specify not to with a `0` or a `-1`.
# 
# 4. Define an [`Integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html#pyblp.Integration) configuration to solve the market share integral from several available options:
# 
#     - Monte Carlo integration (pseudo-random draws).
#     - Product rule quadrature.
#     - Sparse grid quadrature.
# 
# 3. Combine [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation) classes, `product_data`, and the [`Integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html#pyblp.Integration) configuration to construct a [`Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html#pyblp.Problem).
# 4. Use the [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve) method to estimate paramters.
# 
#     - It is required to specify an initial guess of the nonlinear parameters. This serves two primary purposes: speeding up estimation and indicating to the solver through initial values of zero which parameters are restricted to be always zero.

# ## Specification of Random Taste Parameters
# 
# It is common to assume that $f(\beta_i \mid \theta)$ follows a multivariate normal distribution, and to break it up into three parts:
# 
# 1. A mean $K_1 \times 1$ taste which all individuals agree on, $\beta$.
# 2. A $K_2 \times K_2$ covariance matrix, $V$. As is common with multivariate normal distributions, $V$ is not estimated directly. Rather, its matrix square (Cholesky) root $\Sigma$ is estimated where $\Sigma\Sigma' = V$.
# 3. Any $K_2 \times D$ interactions, $\Pi$, with observed $D \times 1$ demographic data, $d_i$.
# 
# Together this gives us that
# 
# $$\beta_i \sim N(\beta + \Pi d_i, \Sigma\Sigma').$$
# 
# [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve) takes an initial guess $\Sigma_0$ of $\Sigma$. It guarantees that $\hat{\Sigma}$ (the estimated parameters) will have the same sparsity structure as $\Sigma_0$. So any zero element of $\Sigma$ is restricted to be zero in the solution $\hat{\Sigma}$. For example, a popular restriction is that $\Sigma$ is diagonal, this can be achieved by passing a diagonal matrix as $\Sigma_0$.

# ## Loading Data
# 
# The `product_data` argument of [`Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html#pyblp.Problem) should be a structured array-like object with fields that store data. Product data can be a structured [NumPy](https://numpy.org/) array, a [pandas](https://pandas.pydata.org/) DataFrame, or other similar objects.

# In[ ]:


product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.head()


# The product data contains `market_ids`, `product_ids`, `firm_ids`, `shares`, `prices`, a number of other firm IDs and product characteristics, and some pre-computed excluded `demand_instruments0`, `demand_instruments1`, and so on. The `product_ids` will be incorporated as fixed effects. 
# 
# For more information about the instruments and the example data as a whole, refer to the [`data`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.data.html#module-pyblp.data) module.

# ## Setting Up and Solving the Problem Without Demographics
# 
# Formulations, product data, and an integration configuration are collectively used to initialize a [`Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html#pyblp.Problem). Once initialized, [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve) runs the estimation routine. The arguments to [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve) configure how estimation is performed. For example, `optimization` and `iteration` arguments configure the optimization and iteration routines that are used by the outer and inner loops of estimation.
# 
# We'll specify [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation) configurations for $X_1$, the demand-side linear characteristics, and $X_2$, the nonlinear characteristics.
# 
# - The formulation for $X_1$ consists of `prices` and fixed effects constructed from `product_ids`, which we will absorb using `absorb` argument of [`Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html#pyblp.Formulation).
# - If we were interested in reporting estimates for each fixed effect, we could replace the formulation for $X_1$ with `Formulation('prices + C(product_ids)')`.
# - Because `sugar`, `mushy`, and the constant are collinear with `product_ids`, we can include them in $X_2$ but not in $X_1$.

# In[ ]:


X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
product_formulations = (X1_formulation, X2_formulation)
product_formulations


# We also need to specify an [`Integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html#pyblp.Integration) configuration. We consider two alternatives:
# 
# 1. Monte Carlo draws: we simulate 50 individuals from a random normal distribution. This is just for simplicity. In practice quasi-Monte Carlo sequences such as Halton draws are preferable, and there should generally be many more simulated individuals than just 50.
# 2. Product rules: we construct nodes and weights according to a product rule that exactly integrates polynomials of degree $5 \times 2 - 1 = 9$ or less.

# In[ ]:


mc_integration = pyblp.Integration('monte_carlo', size=50, specification_options={'seed': 0})
mc_integration


# In[ ]:


pr_integration = pyblp.Integration('product', size=5)
pr_integration


# In[ ]:


mc_problem = pyblp.Problem(product_formulations, product_data, integration=mc_integration)
mc_problem


# In[ ]:


pr_problem = pyblp.Problem(product_formulations, product_data, integration=pr_integration)
pr_problem


# As an illustration of how to configure the optimization routine, we'll use a simpler, non-default [`Optimization`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Optimization.html#pyblp.Optimization) configuration that doesn't support parameter bounds, and use a relatively loose tolerance so the problems are solved quickly. In practice along with more integration draws, it's a good idea to use a tighter termination tolerance.

# In[ ]:


bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-4})
bfgs


# We estimate three versions of the model:
# 
# 1. An unrestricted covariance matrix for random tastes using Monte Carlo integration.
# 2. An unrestricted covariance matrix for random tastes using the product rule.
# 3. A restricted diagonal matrix for random tastes using Monte Carlo Integration.
# 
# Notice that the only thing that changes when we estimate the restricted covariance is our initial guess of $\Sigma_0$. The upper diagonal in this initial guess is ignored because we are optimizing over the lower-triangular Cholesky root of $V = \Sigma\Sigma'$.

# In[ ]:


results1 = mc_problem.solve(sigma=np.ones((4, 4)), optimization=bfgs)
results1


# In[1]:


results2 = pr_problem.solve(sigma=np.ones((4, 4)), optimization=bfgs)
results2


# In[1]:


results3 = mc_problem.solve(sigma=np.eye(4), optimization=bfgs)
results3


# We see that all three models give similar estimates of the price coefficient $\hat{\alpha} \approx -30$. Note a few of the estimated terms on the diagonal of $\Sigma$ are negative. Since the diagonal consists of standard deviations, negative values are unrealistic. When using another optimization routine that supports bounds (like the default L-BFGS-B routine), these diagonal elements are by default bounded from below by zero.

# ## Adding Demographics to the Problem
# 
# To add demographic data we need to make two changes:
# 
# 1. We need to load `agent_data`, which for this cereal problem contains pre-computed Monte Carlo draws and demographics.
# 2. We need to add an `agent_formulation` to the model.
# 
# The `agent data` has several reserved column names.
# 
# - `market_ids` are the index that link the `agent data` to the `market_ids` in `product data`.
# - `weights` are the weights $w_{it}$ attached to each agent. In each market, these should sum to one so that $\sum_i w_{it} = 1$. It is often the case the $w_{it} = 1 / I_t$ where $I_t$ is the number of agents in market $t$, so that each agent gets equal weight. Other possibilities include quadrature nodes and weights.
# - `nodes0`, `nodes1`, and so on are the nodes at which the unobserved agent tastes $\mu_{ijt}$ are evaluated. The nodes should be labeled from $0, \ldots, K_2 - 1$ where $K_2$ is the number of random coefficients.
# - Other fields are the realizations of the demographics $d$ themselves.

# In[1]:


agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
agent_data.head()


# The `agent formulation` tells us which columns of demographic information to interact with $X_2$.

# In[1]:


agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
agent_formulation


# This tells us to include demographic realizations for `income`, `income_squared`, `age`, and the presence of children, `child`, but to ignore other possible demographics when interacting demographics with $X_2$. We should also generally exclude the constant from the demographic formula.
# 
# Now we configure the [`Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html#pyblp.Problem) to include the `agent_formulation` and `agent_data`, which follow the `product_formulations` and `product_data`.
# 
# When we display the class, it lists the demographic interactions in the table of formulations and reports $D = 4$, the dimension of the demographic draws.

# In[1]:


nevo_problem = pyblp.Problem(
    product_formulations,
    product_data,
    agent_formulation,
    agent_data
)
nevo_problem


# The initialized problem can be solved with [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve). We'll use the same starting values as [Nevo (2000a)](https://pyblp.readthedocs.io/en/stable/references.html#nevo-2000a). By passing a diagonal matrix as starting values for $\Sigma$, we're choosing to ignore covariance terms. Similarly, zeros in the starting values for $\Pi$ mean that those parameters will be fixed at zero.
# 
# To replicate common estimates, we'll use the non-default BFGS optimization routine (with a slightly tighter tolerance to avoid getting stuck at a spurious local minimum), and we'll configure [`Problem.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html#pyblp.Problem.solve) to use 1-step GMM instead of 2-step GMM.

# In[1]:


initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
initial_pi = np.array([
  [ 5.4819,  0,      0.2037,  0     ],
  [15.8935, -1.2000, 0,       2.6342],
  [-0.2506,  0,      0.0511,  0     ],
  [ 1.2650,  0,     -0.8091,  0     ]
])
tighter_bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-5})
nevo_results = nevo_problem.solve(
    initial_sigma,
    initial_pi,
    optimization=tighter_bfgs,
    method='1s'
)
nevo_results


# Results are similar to those in the original paper with a (scaled) objective value of $q(\hat{\theta}) = 4.65$ and a price coefficient of $\hat{\alpha} = -62.7$. 

# ## Restricting Parameters
# 
# Because the interactions between `price`, `income`, and `income_squared` are potentially collinear, we might worry that $\hat{\Pi}_{21} = 588$ and  $\hat{\Pi}_{22} = -30.2$ are pushing the price coefficient in opposite directions. Both are large in magnitude but statistically insignficant. One way of dealing with this is to restrict $\Pi_{22} = 0$.
# 
# There are two ways we can do this:
# 
# 1. Change the initial $\Pi_0$ values to make this term zero.
# 2. Change the agent formula to drop `income_squared`.
# 
# First, we'll change the initial $\Pi_0$ values.

# In[1]:


restricted_pi = initial_pi.copy()
restricted_pi[1, 1] = 0
nevo_problem.solve(
    initial_sigma,
    restricted_pi,
    optimization=tighter_bfgs,
    method='1s'
)


# Now we'll drop both `income_squared` and the corresponding column in $\Pi_0$.

# In[1]:


restricted_formulation = pyblp.Formulation('0 + income  + age + child')
deleted_pi = np.delete(initial_pi, 1, axis=1)
restricted_problem = pyblp.Problem(
    product_formulations,
    product_data,
    restricted_formulation,
    agent_data
)
restricted_problem.solve(
    initial_sigma,
    deleted_pi,
    optimization=tighter_bfgs,
    method='1s'
)


# The parameter estimates and standard errors are identical for both approaches. Based on the number of fixed point iterations, there is some evidence that the solver took a slightly different path for each problem, but both restricted problems arrived at identical answers. The $\hat{\Pi}_{12}$ interaction term is still insignificant.
if __name__ == "__main__":
    print("Starting the program...")