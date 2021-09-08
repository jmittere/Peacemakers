# bvm_model.py
# Phase 1
import warnings
from mesa import Model
from mesa import agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from agent_bvm import bvmAgent

CLUSTER_THRESHOLD = .05
EQUILIBRIUM_THRESHOLD = 5

warnings.simplefilter('error', RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def getOpinion(model, agent_num, iss_num):
    # Return the opinion value of a specific agent on a specific issue.
    return model.schedule.agents[agent_num].opinions[iss_num]

def get_avg_assort(model):
    # Compute the average graph assortativity, with respect to opinion values.
    # "Average" means the average over all issues.
    assorts = []
    uniformOpinion = True
    opinion = -1
    for i in range(0, model.num_issues):
        try:
            assort = nx.numeric_assortativity_coefficient(model.G,
                "iss_{}".format(i))
            assorts.append(assort)
        except RuntimeWarning:
            # check if all agents have same opinion
            if isIssueUniform(model, i):
                # Perfect uniformity, so consider this issue to have
                # assortativity 0.
                assorts.append(0)

    return (sum(assorts) / len(assorts))


def returnPersuasionsPerCapita(model):
    return model.persuasions / model.num_agents

def get_mean_opinion_var(model):
    return np.array(
        [np.array([a.opinions[i] for a in model.schedule.agents])
         for i in range(model.num_issues)
         ]).var()

def avg_agt_opinion(x, i, model):
    avg_agt_opinion = [agent.opinions[i] for agent in model.schedule.agents]
    return round(sum(avg_agt_opinion) / len(avg_agt_opinion), 2)


# Return true if all agents have an identical opinion on this issue.
def isIssueUniform(model, issueNum):
    return all([ math.isclose(model.schedule.agents[j].opinions[issueNum],
            model.schedule.agents[0].opinions[issueNum]) for j in range(1, len(model.schedule.agents))])


def getNumClusters(model, issueNum):
    # Return the number of opinion clusters for the issue number passed.
    num_clusters = 0

    clustersList = []
    for i in range(model.num_agents):
        hasBeenAdded = False
        agentOpinion = model.schedule.agents[i].opinions[issueNum]
        if num_clusters == 0:
            # First agent, so make a new list for it.
            cluster = [agentOpinion]
            clustersList.append(cluster)
            hasBeenAdded = True
            num_clusters += 1
        else:
            # Not the first agent, so check if it's within threshold of another
            # cluster.
            for cluster in clustersList:
                avgOpinion = round(sum(cluster) / len(cluster),2)

                if((abs(avgOpinion - agentOpinion) < CLUSTER_THRESHOLD) and
                        (hasBeenAdded == False)):
                    #this agent belongs in this cluster
                    cluster.append(agentOpinion)
                    hasBeenAdded = True

            # If this agent did not belong to any of the existing clusters,
            # make a new cluster for it.
            if(hasBeenAdded == False):
                cluster = [agentOpinion]
                clustersList.append(cluster)
                num_clusters += 1

    return num_clusters


def numNonUniformIssues(model):
    # Returns the number of issues on which all agents agree (within the
    # CLUSTER_THRESHOLD).
    counter = 0
    for i in range(0, model.num_issues):
        if(getNumClusters(model,i)!=1):
            counter+=1
    return counter

def printAllAgentOpinions(model):
    for i in range(model.num_agents):
        for j in range(model.num_issues):
            print("Agent #{}, Issue #{}: {}".format(i, j,
                model.G.nodes[i]["agent"].opinions[j]))

def getPersuasions(model):
    return model.persuasions

def getRepulsions(model):
    return model.repulsions


class bvmModel(Model):

    # N: # of agents
    # P: prob of edge for Erdos-renyi
    # I: # of issues for each agent
    # T: # of simulation iterations
    # C: openness threshold for agents
    # D: disgust threshold for agents
    def __init__(self, l_steps, n_agents, p, issues, c, d, seed=None):
        super().__init__()
        self.l_steps = l_steps
        self.num_agents = n_agents
        self.num_issues = issues
        self.openness_threshold = c
        self.disgust_threshold = d
        self.schedule = RandomActivation(self)
        self.steps = 0
        self.repulsions = 0
        self.persuasions = 0
        self.equilibriumCounter= 0
        self.running = True

        # generate ER graph with N nodes and prob of edge of P
        self.G = nx.erdos_renyi_graph(n_agents, p)
        while not nx.is_connected(self.G):
            self.G = nx.erdos_renyi_graph(n_agents, p)

        # add N number of agents
        for i in range(self.num_agents):
            agent = bvmAgent(i, self)
            self.G.nodes[i]["agent"] = agent
            self.schedule.add(agent)

        reporters =  {"clustersforIssue_{}".format(i):
            lambda model, issueNum=i:
            getNumClusters(model,issueNum) for i in range(self.num_issues)}

        self.datacollector = DataCollector(
            model_reporters=reporters,
            agent_reporters={}
        )

        self.datacollector._new_model_reporter("assortativity", get_avg_assort)
        self.datacollector._new_model_reporter("opinionClusters",
            numNonUniformIssues)
        self.datacollector._new_model_reporter("persuasions", getPersuasions)
        self.datacollector._new_model_reporter("repulsions", getRepulsions)

        self.datacollector.collect(self)


    def step(self):
        self.schedule.step()
        if self.persuasions == 0:
            self.equilibriumCounter += 1
        else:
            # Reset equilibrium counter if there are persuasions still
            # happening.
            self.equilibriumCounter = 0

        self.datacollector.collect(self)

        # Stop if exceeds step limit
        if self.l_steps == self.steps + 1:
            self.running = False

        # Stop if there have been no persuasions for longer than threshold.
        if self.equilibriumCounter > EQUILIBRIUM_THRESHOLD:
            self.running = False

        self.steps += 1
#lsteps, agents, p, issues, othresh, dthresh
test = bvmModel(150, 10, 0.4, 3, 0.20, .70)

#printAllAgentOpinions(test)

for i in range(100):
    test.step()

#printAllAgentOpinions(test)

df = test.datacollector.get_model_vars_dataframe()
print(df)
