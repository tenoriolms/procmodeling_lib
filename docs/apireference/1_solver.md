<h1> EDO Solver </h1>


--------------------------------------------------------------------------------------


# `edo_functions`

```python
class edo_functions()
```

Class for reading, parsing, solving, visualizing, and optimizing systems of ordinary differential equations (ODEs). The system can be loaded from a text file or from a docstring containing mathematical expressions. The class supports numerical integration and parameter optimization.

**Parameters:**

* `name`: `str`

    Path to a text file or a docstring containing the system of equations.


**Default Parameters:**

* `verbose = True`: `bool`

    If True, prints information when reading the equations.


**Attributes:**

* `diff_equations`: `dict`

    Dictionary containing the parsed differential equations.

* `expressions`: `dict`

    Dictionary containing additional expressions used in the system.

* `params`: `dict`

    Dictionary containing the constant parameters of the system.

* `lines`: `list`

    List of strings representing each line read from the input.

* `function`: callable

    The system of ODEs as a callable function compatible with SciPy's `solve_ivp`.
    Takes time `t` and state vector `y`, returns the derivatives `dy/dt` as a list.

* `result`: scipy solution object

    Solution object after solving the system.

* `opt_result`: OptimizeResult

    Result of the optimization.

* `opt_params`: `dict`

    Optimized parameters.


**Methods:**

* `show_equations()`

    Displays the system of equations using LaTeX formatting.

* `_call_function(equations, params)`

    Constructs and returns a callable ODE system function for `solve_ivp` from a list of equation strings and parameter values.

* `solve(t_span, y0, t_eval=None, method='RK45', show_plot=True, _params=None, _data=None, _var_columns=None, _time_column=None)`

    Solves the system of ODEs numerically using `scipy.integrate.solve_ivp`.

* `_plot_results(data, var_columns, time_column)`

    Plots the solution curves and optionally overlays experimental data.

* `optimize(data, initial_values_columns, target_values_columns, time_column, bounds, method='differential_evolution', solver_method='RK45', score='rmse', max_iter_DE=100, popsize_DE=15)`

    Optimizes model parameters to fit experimental data using evolutionary algorithms.

* `_call_objective_function()`

    Builds and returns the objective function used for parameter optimization.

**Returns:**

* `None`

**Notes:**

- The system of equations must be written with each equation in the form `dydt[i] = expression`.
- Additional expressions and parameters can be defined using `key = value` syntax.
- During optimization, each distinct initial condition is automatically identified, and the system is solved for each one.
- Supported score metrics include `rmse`, `r2`, `neg_r2`, `mape`, and `c_coeff`.


**Examples:**

```python
edo = edo_functions('model.txt')
edo.solve(t_span=[0,10], y0=[1,0])
edo.optimize(
    data=df,
    initial_values_columns=['A', 'B'],
    target_values_columns=['A', 'B'],
    time_column='time',
    bounds={'k1': (0, 1), 'k2': (0, 1)}
)
```

**References:**

* [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
* [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)

**Dependencies:**

* numpy
* pandas
* scipy
* matplotlib
* tqdm


---------------------------------------------------------------


## method: `show_equations()`

Displays the differential equations, expressions, and parameters in a formatted LaTeX style plot.

**Examples**

```python
edo = edo_functions('equations.txt')
edo.show_equations()
```

---------------------------------------------------------------

## method: `_call_function()`

Constructs and returns a callable function representing the ODE system, compatible with SciPy's solve_ivp.

The returned function takes time t and state vector y as inputs, evaluates the right-hand side of each equation string using the provided parameters and predefined mathematical functions, and returns the derivatives as a list.

**Parameters:**

- `equations`: `list`  
  List of strings representing the right-hand side expressions of the ODE system.

- `params`: `dict`  
  Dictionary containing parameter names and their numerical values used in the equations.


**Examples**

```python
equations = ['-k1 * y[0]', 'k1 * y[0] - k2 * y[1]']
params = {'k1': 0.1, 'k2': 0.05}
edo_func = edo._call_function(equations, params)
result = solve_ivp(edo_func, (0, 10), [1, 0])
```

---------------------------------------------------------------

## method: `solve()`

Solves the system of ODEs using SciPy's `solve_ivp` and optionally plots the results.

**Parameters:**

- `t_span`: `list`  
  Time span as a list `[t_start, t_end]` for the integration.

- `y0`: `list`  
  List of initial values for the ODE system.

**Default Parameters:**

- `t_eval = None`: `array-like`  
  Optional. Time points at which to store the computed solution.

- `method = 'RK45'`: `str` 
  Integration method to use (e.g., `'RK45'`, `'BDF'`, etc.).

- `show_plot = True`: `bool`   
  Whether to display a plot of the results.

- `_params = None`: `dict` 
  Optional. Dictionary of parameter values to use in the ODE system.

- `_data = None`: `pandas.DataFrame` 
  Optional. Experimental data for comparison with model predictions.

- `_var_columns = None`: `list`  
  Optional. List of variable names in `_data` corresponding to the ODE outputs.

- `_time_column = None`: `str`  
  Optional. Name of the time column in `_data`.

**Examples**

```python
edo = edo_functions('equations.txt')
edo.solve(t_span=[0, 10], y0=[1.0, 0.0])
```

---------------------------------------------------------------

## method: `_plot_results()`

Plots the results of the ODE solution and optionally overlays experimental data for comparison.

**Parameters:**

- `data`: `pandas.DataFrame` or `None`  
  Experimental data to overlay on the simulation results.

- `var_columns`: `list` or `None`  
  List of variable names in `data` corresponding to the ODE outputs.

- `time_column`: `str` or `None`  
  Name of the time column in `data`.

**Examples**

```python
# Assuming `edo` has already solved the ODE system:
edo._plot_results(data=experimental_data, var_columns=['A', 'B'], time_column='time')
```

---------------------------------------------------------------


## method: `optimize()`

Performs parameter optimization to fit the ODE model to experimental data using the specified optimization method.

**Parameters:**

- `data`: `pandas.DataFrame`  
  Experimental dataset containing time series and target variables.

- `initial_values_columns`: `list`  
  List of column names in `data` representing the initial values of the ODE system.

- `target_values_columns`: `list`  
  List of column names in `data` representing the target variables to be fitted.

- `time_column`: `str`  
  Name of the time column in `data`.

- `bounds`: `dict`  
  Dictionary defining the lower and upper bounds for each parameter to be optimized.

**Default Parameters:**

- `method = 'differential_evolution'`: `str`  
  Optimization method to use (currently supports `'differential_evolution'`).

- `solver_method = 'RK45'`: `str`  
  Numerical integration method for solving the ODEs (e.g., `'RK45'`, `'BDF'`).

- `score = 'rmse'`: `str`  
  Evaluation metric to minimize (`'rmse'`, `'r2'`, `'mape'`, `'c_coeff'`, etc.).

- `max_iter_DE = 100`: `int`  
  Maximum number of iterations for the differential evolution optimizer.

- `popsize_DE = 15`: `int`  
  Population size multiplier for the differential evolution optimizer.

**Examples**

```python
edo = edo_functions('equations.txt')

bounds = {'k1': (0, 1), 'k2': (0, 1)}

edo.optimize(
    data=experimental_data,
    initial_values_columns=['A0', 'B0'],
    target_values_columns=['A', 'B'],
    time_column='time',
    bounds=bounds
)
```

---------------------------------------------------------------


## method: `_call_objective_function()`

Builds and returns the objective function used for parameter optimization.

The objective function solves the ODE system for each set of initial conditions found in the provided dataset, compares the simulated results with the experimental data, and computes the cumulative error based on the selected scoring metric (e.g., RMSE).

**Examples**

```python
edo = edo_functions('equations.txt')
objective_function = edo._call_objective_function()
error = objective_function([0.1, 0.05])  # Example parameter values
```