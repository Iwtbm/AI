# Template Code Adapted from AIMA

class BayesNode:
    """
    This class creates a BayesNode object for each node which is part of a BayesNet.

    This class will store the conditional probability distribution for a boolean variable,
    P(X | parents).
    """

    def __init__(self, X, parents, cpt):
        """
        The BayesNode is initialized with the following parameters
        :param X: a variable name (string)
        :param parents: a list of variable names (strings)
        :param cpt: a dictionary corresponding to a conditional probability distribution

        Sample input 1:

        >>> X = 'Burglary'
        >>> parents = [] # no parent
        >>> cpt = 0.001 # true value

        Sample input 2:

        >>> X = 'JohnCalls'
        >>> parents = ['Alarm'] # single parent
        >>> cpt = {True: 0.90, False: 0.05} # T/F values for Alarm

        Sample input 3:

        >>> X = 'Alarm'
        >>> parents = ['Burglary', 'Earthquake'] # two parents
        >>> cpt = {(True, True): 0.95, (True, False): 0.94, (False, True): 0.29, (False, False): 0.001} # T/F values for Burglary/Earthquake
        """

        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def get_parents(self):
        """
        Return the list of parents for the current node
        """
        return self.parents

    def p(self, value, evidence):
        """

        :param value: Boolean value of the node
        :param evidence: Boolean for each parent of the value node
        :return: Returns the requested probability value P(X|evidence)

        Return the conditional probability
        P(X=value | parents=evidence), where evidence
        are the values of parents. (evidence must assign each
        parent a value.)

        Using a node created with Sample Input 3 above:
        >>> node.p(True, {'Burglary': False, 'Earthquake': True})
        >>> 0.29
        """
        assert isinstance(value, bool)
        prob = self.cpt[tuple([evidence[var] for var in self.parents])]
        return prob if value else 1 - prob

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
    """
    This class creates a Bayesian network containing only boolean-variable nodes.
    """

    def __init__(self, node_specs=None):
        """
        For each node in the network create a list of node specifications containing
        the cpt and relationship per node. View the documentation for the specification
        for a node above in the BayesNode class

        NOTE: Nodes must be ordered with parents before children.

        :param node_specs: List of node specifications

        Example call:

        >>> node1 = ('Node1', [], 0.30)
        >>> node2 = ('Node2', ['Node1'], {True: 0.70, False: 0.15})
        >>> bayes_net = BayesNet([node1, node2])

        """
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add_node(node_spec)

    def add_node(self, node_spec):
        """
        Adds a node to the bayes network. Its parents must already be in the
        BayesNet, and its variable must not exist already.

        :param node_spec: a single node specification
        """
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.get_parents())
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.get_parents():
            self.get_node(parent).children.append(node)

    def get_node(self, variable):
        """
        Return the BayesNode node object for the requested variable.

        Example call:

        >>> burglary.get_node('Burglary').variable
        >>> 'Burglary'

        :param variable: String name of the variable in the bayes net you want to request
        :return: a BayesNode object for the requested variable
        """

        for n in self.nodes:
            if n.variable == variable:
                return n
        raise Exception("No such variable: {}".format(variable))

    def get_variable_domain(self, variable):
        """
        Return the domain of variable.

        :param variable: String name of the variable in the bayes net you want to request
        :return: a list with values the variable can take. In this case it will always be
                [True, False] since we are storing a BayesNet that can only have boolean
                nodes.
        """
        for n in self.nodes:
            if n.variable == variable:
                return [True, False]
        raise Exception("No such variable: {}".format(variable))

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)