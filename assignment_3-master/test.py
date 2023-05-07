from submission import *

import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


BayesNet = BayesianModel()

BayesNet.add_node("A")
BayesNet.add_node("B")
BayesNet.add_node("C")
BayesNet.add_node("D")
BayesNet.add_node("E")
BayesNet.add_node("F")

BayesNet.add_edge("A", "D")
BayesNet.add_edge("A", "C")
BayesNet.add_edge("B", "E")
BayesNet.add_edge("D", "E")
BayesNet.add_edge("C", "F")
BayesNet.add_edge("E", "F")

cpd_a = TabularCPD('A', 2, values=[[0.8], [0.2]])
cpd_b = TabularCPD('B', 2, values=[[0.1], [0.9]])
cpd_ca = TabularCPD('C', 2, values=[[0.9, 0.3], \
                [ 0.1, 0.7]], evidence=['A'], evidence_card=[2])
cpd_da = TabularCPD('D', 2, values=[[0.1, 0.7], \
                [0.9, 0.3]], evidence=['A'], evidence_card=[2])
cpd_e_dbe = TabularCPD('E', 2, values=[[0.95, 0.4, 0.45, 0.1], \
                [0.05, 0.6, 0.55, 0.9]], evidence=['D', 'B'], evidence_card=[2, 2])
cpd_f_cef = TabularCPD('F', 2, values=[[0.9, 0.2, 0.4, 0.05], \
                [0.1, 0.8, 0.6, 0.95]], evidence=['C', 'E'], evidence_card=[2, 2])

BayesNet.add_cpds(cpd_a, cpd_b, cpd_ca, cpd_da, cpd_e_dbe, cpd_f_cef)

solver = VariableElimination(BayesNet)
marginal_prob = solver.query(variables=['A'], evidence={'D':0}, joint=False)
prob = marginal_prob['A'].values

solver = VariableElimination(BayesNet)
marginal_prob = solver.query(variables=['D'], evidence={'C':0}, joint=False)
prob = marginal_prob['D'].values

print(prob[0], prob[1])




# power_plant = make_power_plant_net()
# power_plant = set_probability(make_power_plant_net())
# # print(power_plant.cpds[2])
# a = get_gauge_prob(power_plant)
# print(a)

# games_net = get_game_network()
# # pp = MH_sampler(games_net, initial_state=[])
# # # print(pp)
# # n = 0
# # ps = (0,0,0,0,0,0)
# # loop_number = 10
# # for i in range(loop_number):
# #     pp = MH_sampler(games_net, pp)
# #     print(pp, 1)
# #     if pp == ps:
# #         n += 1
# #     ps = pp
# #     # print(ps, 2)
# # print(n)
# # 
# Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(games_net, initial_state=[])
# print(Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count)
