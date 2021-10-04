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
import scipy
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from graspologic.cluster.autogmm import AutoGMMCluster

from agent_bvm import bvmAgent

CLUSTER_THRESHOLD = .05
EQUILIBRIUM_THRESHOLD = 5

warnings.simplefilter('error', RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def getOpinion(model, agent_num, iss_num):
    # Return the opinion value of a specific agent on a specific issue.
    return model.schedule.agents[agent_num].opinions[iss_num]

def getMultimodalityStatisticClone(model):
    return getMultimodalityStatistic(model,
        len(model.schedule.agents[0].opinions))

def getMultimodalityStatisticOneAgreement(model):
    return getMultimodalityStatistic(model, 1)

def getMultimodalityStatisticTwoAgreements(model):
    return getMultimodalityStatistic(model, 2)

def getMultimodalityStatisticAnticlone(model):
    return getMultimodalityStatistic(model,0)

def getMultimodalityStatistic(model, target):
    # Return a statistic estimating the evidence for multi-modality in the
    # number of opinion agreements that each agent has pairwise with every
    # other.
    pwa = getNumPairwiseAgreements(model)

    # For now, use a blunt object: number of anti-clones and clones.
    return sum([ a == target for a in pwa ])

def getNumPairwiseAgreements(model):
    # For every pair of agents in the entire model, determine the number of
    # issues on which they agree. ("agree" means "in the same cluster for that
    # issue.") Return a list (with N-choose-2 elements) of those counts.
    agents = model.schedule.agents
    agreements = np.empty(int(scipy.special.binom(len(agents),2)),dtype=int)
    agreementNum = 0
    for index1 in range(len(agents)):
        for index2 in range(index1+1,len(agents)):
            # TODO: is CLUSTER_THRESHOLD the right thing to use here?
            agreements[agreementNum] = agents[index1].numAgreementsWith(
                agents[index2], CLUSTER_THRESHOLD)
            agreementNum += 1
    return agreements.tolist()

def get_avg_assort(model):
    # Compute the average graph assortativity, with respect to opinion values.
    # "Average" means the average over all issues.
    assorts = []
    uniformOpinion = True
    opinion = -1

    for i in range(0, model.num_issues):
        if isIssueUniform(model, i):
            #Perfect uniformity, so consider this issue to have
            # assortativity of 0. 
            assorts.append(0)
        else:
            try:
                assort = nx.numeric_assortativity_coefficient(model.G,"iss_{}".format(i))
                assorts.append(assort)
            except RuntimeWarning:
                print("Runtime Warning...")
                raise

    return (sum(assorts) / len(assorts))

def getAllOpinions(model, issueNum):
    oList=[] # an array of every agent's opinion for issueNum
    for i in range(model.num_agents):
        oList.append(model.G.nodes[i]["iss_{}".format(issueNum)])
    return oList

def doAutoGMM(model, issueNum):
    oList = getAllOpinions(model,issueNum) #get all opinions for an issue
    opinion_arr = np.array(oList)
    opinion_arr = opinion_arr.reshape(model.num_agents,1)
    clusters = model.autogmm.fit(opinion_arr) #perform AutoGMM algorithm
    #print("There were {} clusters for issue {} in step {} ".format(clusters.n_components, issueNum, model.steps))
    return clusters.n_components_

def returnPersuasionsPerCapita(model):
    return model.persuasions / model.num_agents

def returnLowOpinions(model, issueNum):
    #returns the number of agents within .1 of 0
    agents = model.schedule.agents
    agentCount=0
    for a in agents:
        if a.opinions[issueNum]<=.10:
            agentCount+=1
    return agentCount

def returnHighOpinions(model, issueNum):
    #returns the number of agents within .1 of 1
    agents = model.schedule.agents
    agentCount=0
    for a in agents:
        if a.opinions[issueNum]>=.90:
            agentCount+=1
    return agentCount

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
                avgOpinion = sum(cluster) / len(cluster)

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
    # Returns the number of issues on which all agents don't agree (within the
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
                round(model.G.nodes[i]["agent"].opinions[j],2)))

def getPersuasions(model):
    return model.persuasions

def getRepulsions(model):
    return model.repulsions

def getSteps(model):
    return model.steps

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
        self.influencesLastStep = 0
        self.equilibriumCounter= 0
        self.running = True
        self.clusterTracking = {} #key:(unique_id, issue) 
        
        self.autogmm = AutoGMMCluster()
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
        
        autoGmmReporters = {"autogmmclustersforIssue_{}".format(i):lambda model, issueNum=i: doAutoGMM(model,issueNum) for i in range(self.num_issues)}

        repubs = {"low_iss_{}".format(i):lambda model, issueNum=i: returnLowOpinions(model,issueNum) for i in range(self.num_issues)}
        dems = {"high_iss_{}".format(i):lambda model, issueNum=i: returnHighOpinions(model,issueNum) for i in range(self.num_issues)}
        reporters.update(dems)
        reporters.update(repubs)
        reporters.update(autoGmmReporters)

        self.datacollector = DataCollector(
            model_reporters=reporters,
            agent_reporters={}
        )
         
        self.datacollector._new_model_reporter("Steps", getSteps)
        #self.datacollector._new_model_reporter("assortativity", get_avg_assort)
        #self.datacollector._new_model_reporter("numberOfNonUniformIssues",
        #    numNonUniformIssues)
        self.datacollector._new_model_reporter("persuasions", getPersuasions)
        self.datacollector._new_model_reporter("repulsions", getRepulsions)
        self.datacollector._new_model_reporter("multiModalityStatClone",
            getMultimodalityStatisticClone)
        self.datacollector._new_model_reporter("multiModalityStatAnticlone",
            getMultimodalityStatisticAnticlone)

        self.datacollector._new_model_reporter("multiModalityStatOneAgreement",
            getMultimodalityStatisticOneAgreement)
        self.datacollector._new_model_reporter("multiModalityStatTwoAgreements",
            getMultimodalityStatisticTwoAgreements)

        self.datacollector.collect(self)


    def step(self):
        self.influencesLastStep = 0
        self.schedule.step()

        if self.influencesLastStep == 0:
            self.equilibriumCounter += 1
        else:
            # Reset equilibrium counter if there are persuasions
            #or repulsions still happening.
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
test = bvmModel(1000, 50, 0.3, 3, 0.10, 0.45)

for i in range(test.l_steps):
    test.step()
    if(test.running == False):
        break

#printAllAgentOpinions(test)
df = test.datacollector.get_model_vars_dataframe()
df.to_csv("singleRun.csv")
print(df)
'''
plt.figure()
plt.plot(df['repulsions'], label='repulsions')
plt.plot(df['persuasions'],label='persuasions')
plt.xlabel('Time (steps)')
plt.ylabel('Repulsions & Persuasions')
plt.legend(loc='lower right')
plt.show()
'''
'''
fig, axs = plt.subplots(2, 2)

fig.suptitle('Republicans (Low Opinions) & Democrats (High Opinions)')
axs[0,0].set_title('Opinion 0')
axs[0,0].plot(df['Steps'],df['low_iss_0'], color='red')
axs[0,0].plot(df['Steps'],df['high_iss_0'], color='blue')
axs[1,0].set_title('Opinion 1')
axs[1,0].plot(df['Steps'],df['low_iss_1'], color='red')
axs[1,0].plot(df['Steps'],df['high_iss_1'], color='blue')
axs[0,1].set_title('Opinion 2')
axs[0,1].plot(df['Steps'],df['low_iss_2'], color='red')
axs[0,1].plot(df['Steps'],df['high_iss_2'], color='blue')
axs[1,1].set_title('Opinion 3')
axs[1,1].plot(df['Steps'],df['low_iss_3'], color='red')
axs[1,1].plot(df['Steps'],df['high_iss_3'], color='blue')
'''

'''
#set axis labels
for ax in axs.flat:
    ax.set(xlabel='Steps', ylabel='# of agents')

#Hide x labels and tick labels for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()
fig.tight_layout()
'''

plt.figure()
plt.plot(df['multiModalityStatClone'], label='clones')
plt.plot(df['multiModalityStatAnticlone'], label='anti-clones')
plt.plot(df['multiModalityStatOneAgreement'], label='oneAgreements', color='green')
plt.plot(df['multiModalityStatTwoAgreements'], label='TwoAgreements', color='red')

plt.xlabel('Time (steps)')
plt.ylabel('# clones/anti-clones/1 Agreements/2 Agreements')
plt.legend(loc='best')
plt.show()
