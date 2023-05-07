# CS 6601 Assignment 3: Bayes Nets (Bonus - 10 Points)

In this part of the assignment, you will understand how inferencing works under the hood. In Part 1 of the assignment, you used the `pgmpy` library as a black-box to perform inference. The task for the Bonus is to implement the Variable Elimination algorithm.

## Resources

You will find the following resources helpful for this assignment.

*Udacity Videos:*  
[Lecture 6 on Bayes Nets](https://classroom.udacity.com/courses/ud954/lessons/6381509770/concepts/64119686570923)  

*Textbook:*   
Chapter 14: Probabilistic Reasoning  

## Setup

1. Clone the project repository from Github

   ```
   git clone https://github.gatech.edu/omscs6601/assignment_3_bonus.git
   ```

2. Navigate to `assignment_3_bonus/` directory

3. Activate the environment you created during Assignment 0 

    ```
    conda activate ai_env
    ```
    
    In case you used a different environment name, to list of all environments you have on your machine you can run `conda env list`.

## Submission

Please include all of your own code for submission in `submission.py`.  

**Important: There is a TOTAL submission limit of 5 on Gradescope for this assignment. This means you can submit a maximum of 5 times during the duration of the assignment. Please use your submissions carefully and do not submit until you have thoroughly tested your code locally.**

**If you're at 4 submissions, use your fifth and last submission wisely. The submission marked as ‘Active’ in Gradescope will be the submission counted towards your grade.**

## Restrictions

You are not allowed to use the `pgmpy` library for the Bonus.

# Part 1: Getting Familiar with the Bayesian Network Implementation

In `bayesnet.py` we provide you with an implementation of a `BayesNet` class that builds a Bayesian Network with variables stored as `BayesNode` objects. 

Take a moment to view the provided code `BayesNet` and `BayesNode` classes and understand how they work. Note that the provided code creates a Bayesian network containing only boolean-variable nodes.

Each node in the network is initialized with the following parameters

    :param X: a variable name (string)
    :param parents: a list of variable names (strings)
    :param cpt: a dictionary corresponding to a conditional probability distribution


You can have three types of nodes:

1. A node with no parent.

If you want to create a node `A` with two possible values, where `P(A)` is 70% true, 30% false, you would require the following parameters:

    X = 'A',      # name of the node
    parents = [], # no parent
    cpt = 0.70    # true value    


2. A node with a single parent. 

If you want to set the distribution for `P(A|G)` to be

|  G  |P(A=true given G)|
| ------ | ----- |
|  T   | 0.75|
|  F   | 0.85| 

you require the following parameters:

    X = 'A',
    parents = ['G'], # single parent
    cpt = {True: 0.75, False: 0.85} # T/F values for G when A=True
    
3. A node with two or more parents. 

If you want to set the following distribution for `P(A|G,T)` to be

| G   |  T  |P(A=true given G and T)|
| --- | --- |:----:|
|T|T|0.15|
|T|F|0.6|
|F|T|0.2|
|F|F|0.1|

you require the following parameters:

    X = 'A',
    parents = ['G', 'T'], # two parents
    cpt = { (True , True ): 0.15, 
            (True , False): 0.60, 
            (False, True ): 0.20, 
            (False, False): 0.10} # T/F values for G/T when A=True


To create a Bayeasian Network, you will create a `BayesNet` instance with a list of `node_specs` as input, where the spec for each node is a tuple of three elements `(X, parents, cpt)` as decribed above :

    node1 = ('Node1', [], 0.30)
    node2 = ('Node2', ['Node1'], {True: 0.70, False: 0.15})
    bayes_net = BayesNet([node1, node2])

*NOTE:* Nodes must be ordered with parents before children. For example, because `node1`  is a parent of `node2`, in the list of `node_specs` the `node1` must be ordered before `node2`.

You will not be required to submit any code for this part. However, you must understand how it works as a `BayesNet` object will be provided to you for your Variable Elimination implementation.

# Part 2: Implementing Variable Elimination

For the main exercise, you will implement the Variable Elimination algorithm as decribed in Chapter 14 of the textbook. 

Take the following Bayes Net as an example:

![Screenshot](/img/bayesnet.png)

## 2a: Factors Class

Factors are involved in a joint distribution for the Variable Elimination Algorithm. See Section 14.4.2 for more information on Factors in the textbook. 

For example to calculate the probability that a Burglary occurs given that John calls and Mary calls `P(B|j,m)` we would require 5 factors:

![Screenshot](/img/factors.png)

We have provided a `Factors` class for you. This class will store the variables and conditional probability tables involved with a particular factor. Additionally you will implement the functionality for performing operation on factors required for Variable Elimination such as:
 - Pointwise product of two factors,
 - Summing out a variable from a product of factors,
 - Normalization
    
### Initialization

This class is initialized using two arguments corresponding to a Factor.

The parameters required are the following:

    :param variables: A list of variables involved in the calculation of the probabilities for a factor
    :param cpt: A dictionary representing a conditional probability table involving all variables in a factor

For example, take factor `f3(A,B,E)` from the above example. The “first” element is given by `P(a|b,e)=0.95` and the “last” by `P(¬a|¬b,¬e)=0.999`. This factor would be created with the following variables:

    variables = ['Alarm', 'Burglary', 'Earthquake']
    cpt =
        {(False, False, False): 0.999,
         (False, False, True ): 0.710,
         (False, True , False): 0.060,
         (False, True , True ): 0.050,
         (True , False, False): 0.001,
         (True , False, True ): 0.290,
         (True , True , False): 0.940,
         (True , True , True ): 0.950}

### Factor Pointwise Product

The `Factor.factor_pointwise_product` method performs a pointwise product of two factors and combines their variables. The product must be conducted between the current factor (that is defined with the variables at initialization) and factor stored in the `second_factor` variable which is passed as an argument to this function. Pointwise Product is used to create joint distributions.

The parameters required are the following:

    :param second_factor: A Factor object corresponding to the second factor to multiply with
    :param bayes_net: A bayesnet object
    :return: A new combined Factor object after performing the pointwise product

For example, take the factors `f4(A)` and `f5(A)` corresponding to `P(j|a)`and `P(m|a)`:

![Screenshot](/img/product.png)

The pointwise product would result in a new factor defined by:

    variables = ['Alarm']
    cpt =
        {(False,): 0.0005, (True,): 0.63}

Also take a look at Figure 14.10 for an example in the textbook.

![Screenshot](/img/product2.png)

### Factor Sum Out
The `Factor.factor_sum_out` method sums out a variable from the result of product of factors. For the purposes of this function, the new factor that stores the product is already the current factor. Summing out a variable from a product of factors is done by adding up the submatrices formed by fixing the variable to each of its values in turn. Summing Out is used for marginalization.

    :param hidden_var: The hidden variable (string) over which we will sum over
    :param bayes_net: A bayesnet object
    :return: A new Factor object after performing sum out

For example, take factor `f3(A,B,C)`, in the textbook example from Figure 14.10, where we want to sum out the hidden variable `A`. 

![Screenshot](/img/sumout.png)

After the sum out `f(B,C)` summing over the hidden variable `A`, we would get a new factor defined by:

    variables = ['B', 'C']
    cpt =
        {(False, False): 0.32,
         (False, True ): 0.48,
         (True , False): 0.96,
         (True , True ): 0.24}

### Normalize

The `Factor.normalize` method returns the normalized cpt probabilities. Keep in mind that this
is performed at the very end of Variable Elimination when the factor contains only a single variable.

## 2b: Helper functions

To implement the Variable Elimination algorithm, you will need some additional helper functions. 

### Making Factors

In the Variable Elimination algorithm you are required to make factors. The `make_factor` function returns a `Factor` object for the variable provided in the Bayes Net's joint distribution given the evidence. You will create the "cpt" and "variables" that will be passed to the constructor of `Factor` for making the factors.

The parameters required are the following:

    :param variable: A variable from the Bayes Net (string)
    :param evidence: A dictionary of observed values (evidences) in the network.
    :param bayes_net: A BayesNet object
    :return: A Factor object

*Hint 1:*
Here "variables" refers to a list consisting of the variable itself (passed as input) and the parents minus any variables that are part of the evidence. This can be created by finding the parents of each node and filtering out those that are not part of the evidence.

*Hint 2:*
Here "cpt" created dictionary is the one similar to the original cpt of the node with only rows that agree with the evidence.

*Hint 3:*
You may find it helpful to use `bayes_net.get_node()` and `node.get_parents()`.

Example usage 1:

    variable = 'MaryCalls'
    evidence = {'JohnCalls': True, 'MaryCalls': True}
    factor = make_factor(variable, evidence, bayes_net)
    print(factor.cpt, factor.variables)
    
    Output:
    {(True,): 0.7, (False,): 0.01}, ['Alarm']
    
Example usage 2:

    variable = 'Alarm'
    evidence = {'Burglary': False}
    factor = make_factor(variable, evidence, bayes_net)
    print(factor.cpt, factor.variables)
    
    Output:
    { (False, False): 0.999,
      (False, True ): 0.710,
      (True , False): 0.001,
      (True , True ): 0.290  }, ['Alarm', 'Earthquake']
        
### Pointwise Product
The `pointwise_product` function extends the Pointwise Product operation to more than two
factors, done sequentially in pairs of two with the help of `Factors.factor_pointwise_product()`. Basically, perform Pointwise Product for a list of factors provided and return a new factor.

The parameters required are the following:

    :param factors: A list of Factors
    :param bayes_net: A BayesNet object
    :return: A new Factor object containing the product of multiple factors
    
### Sum Out
The `sum_out` function eliminates the hidden variables from all factors in the factors list provided
by summing over its values. You will need to use both `Factor.factor_sum_out()` and `pointwise_product()` to finally eliminate a particular variable from all factors by summing over its values and return a new factor.

For example, in section 14.4.2 to compute `f6(B,E)` you need to perform three pointwise products before performing a sum out.

![Screenshot](/img/sumout2.png)

The parameters required are the following:

    :param hidden_var: A hidden variable (string) over which we will sum over
    :param factors: A list of Factors
    :param bayes_net: A BayesNet object
    :return: A new Factor object containing the sum out of multiple factors
    
### Check for Hidden Variables
Finally, one last function is needed to implement Variable Elimination algorithm. The `is_hidden_var` function returns a boolean True/False result if the provided variable is hidden or not when querying `P(X|evidence)`

The parameters required are the following:

    :param variable: variable (string) to check if is hidden or not
    :param X: variable (string) for which you want to calculate P(X|evidence)
    :param evidence: list of evidence variables (strings)
    :return: boolean True/False if variable is hidden or not

## 2c: Variable Elimination algorithm

You will now implement the Variable Elimination function as described in the book with help of the `Factors` class implemented above. 

The `VariableElimination` function computes `P(X|evidence)` for the given Bayes Net with Variable Elimination. The algorithm in Figure 14.11 from the textbook will be helpful.

![Screenshot](/img/ve.png)

The parameters required are the following:

    :param X: the query variable (string)
    :param evidence: a dictionary of observed values (evidences) in the network.
    :param bayes_net: a BayesNet object
    :return: a normalized dictionary containing the probabilities (cpt)

You will find the following functions useful:
- `sum_out()`
- `pointwise_product()`
- `is_hidden_var()`
- `make_factor()`
- `Factor.normalize()`

*NOTE:* For this implementation, perform a reverse ordering over the Bayes Net's variables. You might find Python's `reverse()` function useful.

Example call:

    X = 'Burglary'
    evidence = {'JohnCalls': True, 'MaryCalls': True}
    VariableElimination(X, evidence, bayes_net)
    
    Output:
    {False: 0.7158, True: 0.2842}

# Return your name

A simple task to wind down the assignment. Return your name from the function aptly called `return_your_name()`.
