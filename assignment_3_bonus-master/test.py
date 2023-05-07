from bayesnet import *
from submission import *


# nodeB = ('B', [], 0.001)
# nodeE = ('E', [], 0.002)
# nodeA = ('A', ['B', 'E'], {(True , True ): 0.95, (True , False): 0.94, (False, True ): 0.29, (False, False): 0.001})
# nodeJ = ('J', ['A'], {True: 0.9, False: 0.05})
# nodeM = ('M', ['A'], {True: 0.7, False: 0.01})
# bayes_net = BayesNet([nodeB, nodeE, nodeA, nodeJ, nodeM])

# # variable = 'J'
# # evidence = {'J': True, 'M': True}
# # factor = make_factor(variable, evidence, bayes_net)
# # print(factor.variables)
# # print(factor.cpt)

# X = 'B'
# evidence = {'J': True, 'M': True}
# factors = VariableElimination(X, evidence, bayes_net)
# print(factors)

nodeT = ('T', [], 0.2)
nodeFG = ('FG', ['T'], {True: 0.8, False: 0.05})
nodeG = ('G', ['T', 'FG'], {(True , True ): 0.2, (True , False): 0.95, (False, True ): 0.8, (False, False): 0.05})
nodeFA = ('FA', [], 0.15)
nodeA = ('A', ['G', 'FA'], {(True , True ): 0.55, (True , False): 0.9, (False, True ): 0.45, (False, False): 0.1})
bayes_net = BayesNet([nodeT, nodeFG, nodeG, nodeFA, nodeA])

# variable = 'FG'
# # evidence = {'FG': False, 'FA': False, 'A': True}
# evidence = {}
# factor = make_factor(variable, evidence, bayes_net)
# print(factor.variables)
# print(factor.cpt)

# X = 'A'
# evidence = {}
# factors = VariableElimination(X, evidence, bayes_net)
# print(factors)

X = 'T'
evidence = {'FG':False, 'FA':False, 'A':True}
factors = VariableElimination(X, evidence, bayes_net)
print(factors)
# for i in factors:
#     print(i.variables, i.cpt)












# variables = ['A', 'B']
# cpt = {(True, True):0.3, (True, False):0.7, (False, True):0.9, (False, False):0.1}
# AB = Factor(variables, cpt)
# variables = ['B', 'C']
# cpt = {(True, True):0.2, (True, False):0.8, (False, True):0.6, (False, False):0.4}
# BC = Factor(variables, cpt)
# ABC = AB.factor_pointwise_product(BC, bayes_net)
# variables = ['Alarm']
# cpt = {(False,): 0.0005, (True,): 0.63}
# alarm = Factor(variables, cpt)
# print(alarm.normalize())
# print(ABC.cpt[(True, True, True)])
# print(ABC.factor_sum_out('A',bayes_net).cpt)
