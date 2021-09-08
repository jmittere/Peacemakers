# bvm_model.py
# Phase 1
import warnings
from mesa import Model
from mesa import agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx

from agent_bvm import bvmAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter('error', RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def getOpinion(model, agent_num, iss_num):
    return model.schedule.agents[agent_num].opinions[iss_num]

def get_avg_assort(model):
    assorts = []
    uniformOpinion = True
    opinion = -1
    for i in range(0, model.num_issues):
        try:
            assort = nx.numeric_assortativity_coefficient(model.G, "iss_{}".format(i))
            assorts.append(assort)
        except RuntimeWarning:
            # check if all agents have same opinion
            uniformOpinion = True
            opinion = -1
            for j in range(0, model.num_agents):
                if (opinion == -1):
                    opinion = model.schedule.agents[j].opinions[i]
                else:
                    if (opinion != model.schedule.agents[j].opinions[i]):
                        # opinion is not the same
                        uniformOpinion = False
            if (uniformOpinion):
                # perfect uniformity
                assort = 0
                #print("Added 0 assort at step_{}".format(model.steps))
                assorts.append(assort)

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


def checkissueUniformity(model, issueNum):
    # checks to see if a certain issue is uniform
    opinion = -1
    for i in range(0, model.num_agents):
        if (opinion == -1):
            opinion = model.schedule.agents[i].opinions[issueNum]
        else:
            if (opinion != model.schedule.agents[i].opinions[issueNum]):
                return False

    return True

def getClustering(model, issueNum):
    '''takes a model and the issue to get the clustering for as parameters'''
    if(issueNum >= model.num_issues):
        print("There is no issue {} for this model. num_issues is {}".format(issueNum, model.num_issues))
        return None
    num_clusters = 0

    clustersList = []
    for i in range(model.num_agents):
        hasBeenAdded = False
        agentOpinion = model.schedule.agents[i].opinions[issueNum]
        if num_clusters == 0:  #first agent, make new list for them
            cluster = [agentOpinion]
            clustersList.append(cluster)
            hasBeenAdded = True
            num_clusters += 1
        else:  #not the first agent, check if within threshold of another cluster
            for cluster in clustersList:
                avgOpinion = round(sum(cluster) / len(cluster),2)
                
                if((abs(avgOpinion - agentOpinion) < .05) and (hasBeenAdded == False)):
                    #this agent belongs in this cluster
                    cluster.append(agentOpinion)
                    hasBeenAdded = True

            #the agent did not belong to any of the existing clusters
            #make a new cluster for this agent
            if(hasBeenAdded == False):
                cluster = [agentOpinion]
                clustersList.append(cluster)
                num_clusters += 1
    #print(clustersList) 
    return num_clusters 

def returnNonUniform(model):
    '''returns the number of nonUniformIssues'''
    counter = 0
    for i in range(0, model.num_issues):
        if(getClustering(model,i)!=1):
            counter+=1

    return counter

def printAllAgentOpinions(model): 
    for i in range(model.num_agents):
        for j in range(model.num_issues):
            print("Agent #{}, Issue #{}: {}".format(i, j, model.G.nodes[i]["agent"].opinions[j]))

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

        reporters =  {"clustersforIssue_{}".format(i): lambda model, issueNum=i:getClustering(model,issueNum) for i in range(self.num_issues)}
        
        self.datacollector = DataCollector(

                model_reporters=reporters,
            agent_reporters={}
        )
        
        self.datacollector._new_model_reporter("assortativity", get_avg_assort)
        self.datacollector._new_model_reporter("opinionClusters", returnNonUniform)
        self.datacollector._new_model_reporter("persuasions", getPersuasions)
        self.datacollector._new_model_reporter("repulsions", getRepulsions)


        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        if self.persuasions == 0:
            self.equilibriumCounter += 1
        else:
            # reset equilibrium counter if there is persuasions still happening
            self.equilibriumCounter = 0

        self.datacollector.collect(self)

        # Stop if exceeds step limit
        if self.l_steps == self.steps + 1:
            self.running = False

        # Stop if there has been 0 persuasions for more than 5 consecutive steps
        if self.equilibriumCounter > 5:
            self.running = False

        self.steps += 1
#lsteps, agents, p, issues, othresh, dthresh
test = bvmModel(150, 100, 0.4, 3, 0.30, .50)

#printAllAgentOpinions(test)

for i in range(100):
    test.step()

#printAllAgentOpinions(test)

df = test.datacollector.get_model_vars_dataframe()
print(df)
