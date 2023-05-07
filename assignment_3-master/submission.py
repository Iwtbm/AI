import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function    
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty alarm", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    # raise NotImplementedError    
    cpd_t = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    cpd_fa = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    cpd_fg_t = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], \
                    [ 0.05, 0.8]], evidence=['temperature'], evidence_card=[2])
    cpd_g_tfg = TabularCPD('gauge', 2, values=[[0.95, 0.2, 0.05, 0.8], \
                    [0.05, 0.8, 0.95, 0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])
    cpd_a_gfa = TabularCPD('alarm', 2, values=[[0.9, 0.55, 0.1, 0.45], \
                    [0.1, 0.45, 0.9, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])

    bayes_net.add_cpds(cpd_t, cpd_fa, cpd_fg_t, cpd_g_tfg, cpd_a_gfa)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    prob = marginal_prob['alarm'].values
    alarm_prob = prob[1]
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    prob = marginal_prob['gauge'].values
    gauge_prob = prob[1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'faulty gauge':0, 'faulty alarm':0, 'alarm':1}, joint=False)
    prob = conditional_prob['temperature'].values
    temp_prob = prob[1]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    # raise NotImplementedError    
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")

    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")

    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.3], [0.1]])

    cpd_AvB = TabularCPD('AvB', 3, [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                    [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]], \
                                    evidence=['A', 'B'], evidence_card=[4,4])
    
    cpd_BvC = TabularCPD('BvC', 3, [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                    [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]], \
                                    evidence=['B', 'C'], evidence_card=[4,4])
    
    cpd_CvA = TabularCPD('CvA', 3, [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1], \
                                    [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]], \
                                    evidence=['C', 'A'], evidence_card=[4,4])

    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB':0, 'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    # TODO: finish this function
    # raise NotImplementedError
    if len(sample) == 0:
        initial_state = zeros(6)
        for i in range(3):
            initial_state[i] = random.randint(0, 3)
        initial_state[3] = 0
        initial_state[4] = random.randint(0, 2)
        initial_state[5] = 2
        sample = tuple(initial_state.astype(int))
    else:
        initial_state = list(initial_state)
        for i in range(6):
            if initial_state[i] == None and i < 3:
                initial_state[i] = random.randint(0, 3)          
            elif initial_state[i] == None and i == 3:
                initial_state[3] = 0
            elif initial_state[i] == None and i == 4: 
                initial_state[4] = random.randint(0, 2)
            elif initial_state[i] == None and i == 5: 
                initial_state[5] = 2
            else:
                continue
        sample = tuple(initial_state)

    i = random.randint(0, 3)
    A_cpd = bayes_net.get_cpds("A")
    B_cpd = bayes_net.get_cpds("B")
    C_cpd = bayes_net.get_cpds("C")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    BvC_cpd = bayes_net.get_cpds("BvC")
    CvA_cpd = bayes_net.get_cpds("CvA")

    if i == 0:
        nummerator_list = []
        denominator = 0
        for v in range(len(A_cpd.values)):
            nummerator = A_cpd.values[v] * B_cpd.values[sample[1]] * C_cpd.values[sample[2]] * \
                AvB_cpd.values[sample[3], v, sample[1]] * BvC_cpd.values[sample[4], sample[1], sample[2]] * CvA_cpd.values[sample[5], sample[2], v]
            nummerator_list.append(nummerator)

            denominator = denominator + nummerator
        sample_weights = np.array(nummerator_list) / denominator
        new_sample = list(sample)
        new_sample[0] = random.choices([0,1,2,3], weights=sample_weights)[0]
        sample = tuple(new_sample)

    elif i == 1:
        nummerator_list = []
        denominator = 0
        for v in range(len(B_cpd.values)):
            nummerator = A_cpd.values[sample[0]] * B_cpd.values[v] * C_cpd.values[sample[2]] * \
                AvB_cpd.values[sample[3], sample[0], v] * BvC_cpd.values[sample[4], v, sample[2]] * CvA_cpd.values[sample[5], sample[2], sample[0]]
            nummerator_list.append(nummerator)

            denominator = denominator + nummerator
        sample_weights = np.array(nummerator_list) / denominator
        new_sample = list(sample)
        new_sample[1] = random.choices([0,1,2,3], weights=sample_weights)[0]
        sample = tuple(new_sample)

    elif i == 2:
        nummerator_list = []
        denominator = 0
        for v in range(len(C_cpd.values)):
            nummerator = A_cpd.values[sample[0]] * B_cpd.values[sample[1]] * C_cpd.values[v] * \
                AvB_cpd.values[sample[3], sample[0], sample[1]] * BvC_cpd.values[sample[4], sample[1], v] * CvA_cpd.values[sample[5], v, sample[0]]
            nummerator_list.append(nummerator)

            denominator = denominator + nummerator
        sample_weights = np.array(nummerator_list) / denominator
        new_sample = list(sample)
        new_sample[2] = random.choices([0,1,2,3], weights=sample_weights)[0]
        sample = tuple(new_sample)

    else:
        nummerator_list = []
        denominator = 0
        BvC_results = BvC_cpd.values[:, sample[1], sample[2]]
        for v in range(len(BvC_results)):
            nummerator = A_cpd.values[sample[0]] * B_cpd.values[sample[1]] * C_cpd.values[sample[2]] * \
                AvB_cpd.values[sample[3], sample[0], sample[1]] * BvC_cpd.values[v, sample[1], sample[2]] * CvA_cpd.values[sample[5], sample[2], sample[0]]
            nummerator_list.append(nummerator)

            denominator = denominator + nummerator
        sample_weights = np.array(nummerator_list) / denominator
        new_sample = list(sample)
        new_sample[4] = random.choices([0,1,2], weights=sample_weights)[0]
        sample = tuple(new_sample)

    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function
    # raise NotImplementedError   
    if len(sample) == 0:
        initial_state = zeros(6)
        for i in range(3):
            initial_state[i] = random.randint(0, 3)
        initial_state[3] = 0
        initial_state[4] = random.randint(0, 2)
        initial_state[5] = 2
        sample = tuple(initial_state.astype(int))
    else:
        initial_state = list(initial_state)
        for i in range(6):
            if initial_state[i] == None and i < 3:
                initial_state[i] = random.randint(0, 3)          
            elif initial_state[i] == None and i == 3:
                initial_state[3] = 0
            elif initial_state[i] == None and i == 4: 
                initial_state[4] = random.randint(0, 2)
            elif initial_state[i] == None and i == 5: 
                initial_state[5] = 2
            else:
                continue
        sample = tuple(initial_state) 

    A_cpd = bayes_net.get_cpds("A")
    B_cpd = bayes_net.get_cpds("B")
    C_cpd = bayes_net.get_cpds("C")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    BvC_cpd = bayes_net.get_cpds("BvC")
    CvA_cpd = bayes_net.get_cpds("CvA")
    
    new_sample = zeros(6)
    for i in range(3):
        new_sample[i] = random.randint(0, 3)
    new_sample[3] = 0
    new_sample[4] = random.randint(0, 2)
    new_sample[5] = 2
    new_sample = new_sample.astype(int)

    p_org = A_cpd.values[sample[0]] * B_cpd.values[sample[1]] * C_cpd.values[sample[2]] * \
            AvB_cpd.values[sample[3], sample[0], sample[1]] * BvC_cpd.values[sample[4], sample[1], sample[2]] * CvA_cpd.values[sample[5], sample[2], sample[0]]
    p_new = A_cpd.values[new_sample[0]] * B_cpd.values[new_sample[1]] * C_cpd.values[new_sample[2]] * \
            AvB_cpd.values[new_sample[3], new_sample[0], new_sample[1]] * BvC_cpd.values[new_sample[4], new_sample[1], new_sample[2]] * CvA_cpd.values[new_sample[5], new_sample[2], new_sample[0]]

    r = min(1, p_new/p_org)
    u = random.random()  

    if u <= r:
        sample = tuple(new_sample)
        # print(1)
        
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    # raise NotImplementedError  
    e = 1  
    delta = 0.00001    
    n1 = 0
    n2 = 0
    n3 = 0
    sample = initial_state
    N_g = 1000
    N_MH = 1000
    # for i in range(100000):
    #     sample = Gibbs_sampler(bayes_net, sample)
    #     if i >= 100:
    #         if sample[4] == 0:
    #             n1 += 1
    #         elif sample[4] == 1:
    #             n2 += 1
    #         else:
    #             n3 += 1
    #         Gibbs_count += 1
    # Gibbs_convergence = [n1/Gibbs_count, n2/Gibbs_count, n3/Gibbs_count]
    for i in range(N_g):
        p_conv = Gibbs_convergence
        sample = Gibbs_sampler(bayes_net, sample)
        if sample[4] == 0:
            n1 += 1
        elif sample[4] == 1:
            n2 += 1
        else:
            n3 += 1
        Gibbs_count += 1
    Gibbs_convergence = [n1/Gibbs_count, n2/Gibbs_count, n3/Gibbs_count]
    e = sum(abs(np.array(p_conv) - np.array(Gibbs_convergence)))/3
    
    while e > delta:
        p_conv = Gibbs_convergence
        sample = Gibbs_sampler(bayes_net, sample)
        if sample[4] == 0:
            n1 += 1
        elif sample[4] == 1:
            n2 += 1
        else:
            n3 += 1
        Gibbs_count += 1
        Gibbs_convergence = [n1/Gibbs_count, n2/Gibbs_count, n3/Gibbs_count]
        e = sum(abs(np.array(p_conv) - np.array(Gibbs_convergence)))/3

    e = 1  
    n1 = 0
    n2 = 0
    n3 = 0
    sample = initial_state
    p_sample = sample
    for i in range(100000):
        sample = MH_sampler(bayes_net, sample)
    #     if i >= 100:
    #         if sample == p_sample:
    #             MH_rejection_count += 1
    #         if sample[4] == 0:
    #             n1 += 1
    #         elif sample[4] == 1:
    #             n2 += 1
    #         else:
    #             n3 += 1
    #         MH_count += 1
    #     p_sample = sample

    # MH_convergence = [n1/MH_count, n2/MH_count, n3/MH_count]
    for i in range(N_MH):
        sample = MH_sampler(bayes_net, sample)
        if sample == p_sample:
            MH_rejection_count += 1
        if sample[4] == 0:
            n1 += 1
        elif sample[4] == 1:
            n2 += 1
        else:
            n3 += 1
        MH_count += 1
        p_sample = sample

    MH_convergence = [n1/MH_count, n2/MH_count, n3/MH_count]

    while e > delta:
        p_conv = MH_convergence
        sample = MH_sampler(bayes_net, sample)
        if sample == p_sample:
            MH_rejection_count += 1
        if sample[4] == 0:
            n1 += 1
        elif sample[4] == 1:
            n2 += 1
        else:
            n3 += 1
        MH_count += 1
        MH_convergence = [n1/MH_count, n2/MH_count, n3/MH_count]
        e = sum(abs(np.array(p_conv) - np.array(MH_convergence)))/3
        p_sample = sample

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    # raise NotImplementedError
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.5
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Wenda Xu"
    raise NotImplementedError
