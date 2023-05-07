from bayesnet import *
import numpy as np

class Factor:
    """
    This class corresponds to a Factor involved in a joint distribution
    for the Variable Elimination Algorithm.

    This class will store the variables and conditional probability tables
    involved with a particular factor. Additionally it will provide functionality
    for performing operation on factors required for Variable Elimination
    such as:
        - Pointwise product of two factors,
        - Summing out a variable from a product of factors,
        - Normalization

    See Section 14.4.2 for more information on Factors in the textbook
    """

    def __init__(self, variables, cpt):
        """
        This class is initialized using two arguments corresponding to a Factor
        :param variables: A list of variables involved in the calculation of the probabilities for a factor
        :param cpt: A dictionary representing a conditional probability table involving all variables in a factor

        For example, take factor f3(A,B,E) in the textbook example in Section 14.4.2:

        variables = ['Alarm', 'Burglary', 'Earthquake']
        cpt =
            {(False, False, False): 0.999,
             (False, False, True): 0.71,
             (False, True, False): 0.06,
             (False, True, True): 0.05,
             (True, False, False): 0.001,
             (True, False, True): 0.29,
             (True, True, False): 0.94,
             (True, True, True): 0.95}
        """
        self.variables = variables
        self.cpt = cpt

    def factor_pointwise_product(self, second_factor, bayes_net):
        """
        This method performs a pointwise product of two factors and combines their variables.
        The product must be conducted between the current factor (self) and factor stored in
        the second_factor variable. Pointwise Product is used to create joint distributions.

        :param second_factor: A Factor object corresponding to the second factor to multiply with
        :param bayes_net: A BayesNet object
        :return: A new combined Factor object after performing the pointwise product

        For example, take factor f4(A) and f5(A) in the textbook example in Section 14.4.2.
        The pointwise product would result in a factor with:

        variables = ['Alarm']
        cpt =
            {(False,): 0.0005, (True,): 0.63}

        Also take a look at Figure 14.10 as an example in the textbook
        """
        cpt = {}
        variables = []

        ############################
        ### TODO: YOUR CODE HERE ###
        set1 = set(self.variables)
        set2 = set(second_factor.variables)

        if len(self.variables) == 1 and len(second_factor.variables) == 1:
            variables = self.variables
            cpt[(True,)] = self.cpt[(True,)] * second_factor.cpt[(True,)]
            cpt[(False,)] = self.cpt[(False,)] * second_factor.cpt[(False,)]
            
        else:
            middle_v = list(set1&set2)
            second_factor_v = second_factor.variables.copy()
            index_1 = []
            index_2 = []
            for v in middle_v:
                idx1 = self.variables.index(v)
                idx2 = second_factor.variables.index(v)
                second_factor_v.remove(second_factor.variables[idx2])
                index_1.append(idx1)
                index_2.append(idx2)

            variables = self.variables + second_factor_v

            for i in self.cpt:
                for j in second_factor.cpt:
                    add = True
                    for k in range(len(index_1)):
                        if i[index_1[k]] != j[index_2[k]]:
                            add = False
                            break
                    if add:
                        first_cpt = list(i)
                        second_cpt = list(j)
                        for k in range(len(index_1)):
                            second_cpt.remove(j[k])
                        final_cpt = tuple(first_cpt + second_cpt)
                        cpt[final_cpt] = self.cpt[i] * second_factor.cpt[j]
                

        ### END OF STUDENT CODE ####
        ############################

        return Factor(variables, cpt)

    def factor_sum_out(self, hidden_var, bayes_net):
        """
        This method sums out a variable from a product of factors contained in self. Summing
        out a variable from a product of factors is done by adding up the submatrices formed
        by fixing the variable to each of its values in turn. Summing Out is used for
        marginalization.

        :param hidden_var: The hidden variable (string) over which we will sum over
        :param bayes_net: A BayesNet object
        :return: A new Factor object after performing sum out

        For example, take factor f3(A,B,C) in the textbook example from Figure 14.10.
        After the sum out f(B,C) summing over the hidden variable A, we would get a factor with:

        variables = ['B', 'C']
        cpt =
            {(False, False): 0.32,
             (False, True): 0.48,
             (True, False): 0.96,
             (True, True): 0.24}
        """
        cpt = {}
        variables = []

        ############################
        ### TODO: YOUR CODE HERE ###         
        index = self.variables.index(hidden_var)
        variables = self.variables
        del variables[index]

        for cp in self.cpt:
            ncp = list(cp)
            ncp[index] = not ncp[index]
            ncp = tuple(ncp)
            p_value = self.cpt[cp] + self.cpt[ncp]
            ncp = list(ncp)
            del ncp[index]
            ncp = tuple(ncp)
            cpt[ncp] = p_value
        ### END OF STUDENT CODE ####
        ############################

        return Factor(variables, cpt)

    def normalize(self):
        """
        This method returns the normalized cpt probabilities. Keep in mind that this
        is performed at the very end of Variable Elimination when the factor contains
        only a single variable.
        :return: A normalized conditional probability table (cpt) dictionary
        """
        assert len(self.variables) == 1
        cpt = {}

        ############################
        ### TODO: YOUR CODE HERE ###
        p_true = self.cpt[(True,)]/(self.cpt[(True,)] + self.cpt[(False,)])
        p_false = self.cpt[(False,)]/(self.cpt[(True,)] + self.cpt[(False,)])
        cpt[True] = p_true
        cpt[False] = p_false
        ### END OF STUDENT CODE ####
        ############################

        return cpt

def make_factor(variable, evidence, bayes_net):
    """
    This function returns a Factor object for the variable provided in the
    Bayes Net's joint distribution given the evidence. You will create the
    cpt and variables that will be passed to the constructor of Factor.
    :param variable: A variable from the Bayes Net (string)
    :param evidence: A dictionary of observed values (evidences) in the network.
    :param bayes_net: A BayesNet object
    :return: A Factor object

    Example usage:
        >>> variable = 'MaryCalls'
        >>> evidence = {'JohnCalls': True, 'MaryCalls': True}
        >>> factor = make_factor(variable, evidence, bayes_net)
        >>> print(factor.cpt, factor.variables)
        >>> {(True,): 0.7, (False,): 0.01}, ['Alarm']

    Hint 1:
    Here "variables" refers to a list consisting of the variable itself and
    the parents minus any variables that are part of the evidence. This can
    be created by finding the parents of each node and filtering out those
    that are not part of the evidence.

    Hint 2:
    Here "cpt" created dictionary is the one similar to the original cpt of
    the node with only rows that agree with the evidence.

    Hint 3:
    You may find it helpful to use bayes_net.get_node() and node.get_parents()
    """

    cpt = {}
    variables = []

    ############################
    ### TODO: YOUR CODE HERE ###
    node = bayes_net.get_node(variable)
    e = list(evidence.keys())
    parents = node.get_parents()

    full_list = [variable]
    full_list += parents
    set1 = set(full_list)
    set2 = set(e)
    minus = list(set1&set2)
    index = []
    variables = full_list.copy()
    for i in minus:
        idx = full_list.index(i)
        index.append(idx)
        variables.remove(i)

    old_cpt = node.cpt.copy()
    keys = np.array(list(old_cpt.keys()))

    rows = keys.shape[0]
    v_p = np.hstack((np.ones((1, rows), dtype=bool), np.zeros((1, rows), dtype=bool))).T
    keys = np.vstack((keys, keys))

    keys = np.hstack((v_p, keys)).astype(bool)

    values = np.array(list(old_cpt.values()))
    values = np.hstack((values, 1-values))

    if len(index) != 0:
        index_r = []
        for i in index:
            for k in range(keys.shape[0]):
                if keys[k][i] != evidence[full_list[i]]:
                    index_r.append(k)
        index_r = list(set(index_r))
        keys = np.delete(keys, index_r, axis=0)
        keys = np.delete(keys, index, axis=1)
        values = np.delete(values, index_r)

    for k in range(keys.shape[0]):
        key = tuple(keys[k])
        cpt[key] = values[k]

    ### END OF STUDENT CODE ####
    ############################

    return Factor(variables, cpt)

def pointwise_product(factors, bayes_net):
    """
    This function extends the Pointwise Product operation to more than two
    factors, done sequentially in pairs of two with the help of
    Factors.factor_pointwise_product(). Basically, perform Pointwise Product
    for a list of factors provided and return a new factor.
    :param factors: A list of Factors
    :param bayes_net: A BayesNet object
    :return: A new Factor object containing the product of multiple factors
    """
    factor = None

    ############################
    ### TODO: YOUR CODE HERE ###
    factor = factors[0]
    for i in factors:
        if factor == i:
            continue

        factor = factor.factor_pointwise_product(i, bayes_net)
    ### END OF STUDENT CODE ####
    ############################

    return factor

def sum_out(hidden_var, factors, bayes_net):
    """
    This function eliminates the hidden_var from all factors in the factors list provided
    by summing over its values. You will need to use both Factor.factor_sum_out() and
    pointwise_product() to finally eliminate a particular variable from all factors by
    summing over its values and return a new factor.

    For example, in section 14.4.2 to compute f6(B, E) you need to perform three pointwise
    products before performing a sum out.

    :param hidden_var: A hidden variable (string) over which we will sum over
    :param factors: A list of Factors
    :param bayes_net: A BayesNet object
    :return: A new Factor object containing the sum out of multiple factors
    """
    factor = None

    ############################
    ### TODO: YOUR CODE HERE ###
    p_factor = pointwise_product(factors, bayes_net)
    factor = p_factor.factor_sum_out(hidden_var, bayes_net)
    ### END OF STUDENT CODE ####
    ############################

    return factor

def is_hidden_var(variable, X, evidence):
    """
    This function returns a boolean True/False result if the provided variable
    is hidden or not when querying P(X|evidence)
    :param variable: variable (string) to check if is hidden or not
    :param X: variable (string) for which you want to calculate P(X|evidence)
    :param evidence: list of evidence variables (strings)
    :return: boolean True/False if variable is hidden or not
    """
    is_hidden = None

    ############################
    ### TODO: YOUR CODE HERE ###
    if variable not in evidence and variable != X:
        is_hidden = True
    else:
        is_hidden = False
    ### END OF STUDENT CODE ####
    ############################

    return is_hidden

def VariableElimination(X, evidence, bayes_net):
    """
    This function computes P(X|evidence) for the given Bayes Net with Variable Elimination
    The algorithm in Figure 14.11 from the textbook will be helpful.
    :param X: the query variable (string)
    :param evidence: a dictionary of observed values (evidences) in the network.
    :param bayes_net: a BayesNet object
    :return: a normalized dictionary containing the probabilities (cpt)

    You will find the following functions useful:
        - sum_out()
        - pointwise_product()
        - is_hidden_var()
        - make_factor()
        - Factor.normalize()

    NOTE:   For this implementation, perform a reverse ordering over the Bayes Net's variables.
            You might find Python's reverse() function useful.

    Example call:

        >>> X = 'Burglary'
        >>> evidence = {'JohnCalls': True, 'MaryCalls': True}
        >>> VariableElimination(X, evidence, bayes_net)
        >>> {False: 0.7158, True: 0.2842}

    """
    cpt = {}

    ############################
    ### TODO: YOUR CODE HERE ###
    factors = []
    order = bayes_net.nodes
    order.reverse()
    for i in order:
        variable = i.variable
        f = make_factor(variable, evidence, bayes_net)
        if len(f.variables) == 0:
            continue
        factors.append(f)

        if is_hidden_var(variable, X, evidence):
            f = sum_out(variable, factors, bayes_net)
            factors = [f]

    factor = pointwise_product(factors, bayes_net)


    cpt = factor.normalize()

    ### END OF STUDENT CODE ####
    ############################

    return cpt

def return_your_name():
    """
    :return: Return your name
    """
    return "Wenda Xu"