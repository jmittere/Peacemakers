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

def returnOptimalK(scores):
    highest = -2
    optimalK = -1
    #scores[0] is 2 clusters, NOT 0 clusters
    #print("Silhouettes: ", scores)
    for i in range(len(scores)):
        if(scores[i]>highest):
            highest = scores[i]
            optimalK = i

    #print("Silhouette Avgs max: ", max(scores))
    return optimalK + 2  

def plotElbow(k, sums):
    plt.plot(k,sums)
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k value')
    plt.show()

def doKMeans(model, issueNum):
    klist=[] # an array of every agent's opinion for issueNum
    for i in range(model.num_agents):
        klist.append(model.G.nodes[i]["iss_{}".format(issueNum)])
    
    '''
    #plot the agents opinions
    ylist = [0 for i in range(len(klist))]
    plt.scatter(klist, ylist, alpha=0.5)
    plt.xticks(np.arange(0,100,10))
    #plt.show()
    '''

    K = range(2,10) #range of clusters to try for kmeans
    silhouette_avgs = [] 
    sum_squared_distances = []
    opinionList = np.array(klist).reshape(-1,1) #why do I need to do this
    
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit_predict(opinionList) #do the kmeans clustering
        
        assert(len(kmeans.labels_)>1),"less than 2 labels"
        #print("Steps: ", model.steps)
        #print("Labels: ", kmeans.labels_)
        score = silhouette_score(opinionList, kmeans.labels_, metric='euclidean')
        silhouette_avgs.append(score)
        sum_squared_distances.append(kmeans.inertia_)

    #Elbow Method analysis plot
    #plotElbow(K, sum_squared_distances)

    #find the optimal number of clusters with silhouette method
    optimalK = returnOptimalK(silhouette_avgs)
    #print("Optimal K: ", optimalK)
    
    #do KMeans clustering again with the optimized number of clusters
    optimizedKMeans = KMeans(n_clusters=optimalK)
    optimizedKMeans.fit_predict(opinionList)

    #print("Cluster_Centers: ", optimizedKMeans.cluster_centers_)
    #print("Labels: ", optimizedKMeans.labels_)
    
    for i in range(model.num_agents):
        model.clusterTracking[(i, issueNum)] = optimizedKMeans.labels_[i]        


def returnPersuasionsPerCapita(model):
    return model.persuasions / model.num_agents

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
        self.datacollector._new_model_reporter("numberOfNonUniformIssues",
            numNonUniformIssues)
        self.datacollector._new_model_reporter("persuasions", getPersuasions)
        self.datacollector._new_model_reporter("repulsions", getRepulsions)
        self.datacollector._new_model_reporter("multiModalityStatClone",
            getMultimodalityStatisticClone)
        self.datacollector._new_model_reporter("multiModalityStatAnticlone",
            getMultimodalityStatisticAnticlone)

        self.datacollector.collect(self)


    def step(self):
        self.influencesLastStep = 0
        self.schedule.step()
        
        for i in range(self.num_issues):
            with warnings.catch_warnings(): #ignoring the ConvergenceWarning from doKMeans(), can we catch this somehow??
                warnings.simplefilter("ignore")
                #doKMeans(self, i)

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
test = bvmModel(150, 50, 0.40, 3, 0.10, 0.50)
#printAllAgentOpinions(test)

for i in range(test.l_steps):
    if(test.running):
        test.step()
        if(i==25):
            for i in range(test.num_issues):
                #doKMeans(test, i)
                pass
            #print(test.clusterTracking)
    else:
        break
#printAllAgentOpinions(test)

df = test.datacollector.get_model_vars_dataframe()
print(df)
plt.plot(df['repulsions'], label='repulsions')
plt.plot(df['persuasions'],label='persuasions')

plt.xlabel('Time (steps)')
plt.ylabel('Repulsions & Persuasions')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(df['multiModalityStatClone'], label='clones')
plt.plot(df['multiModalityStatAnticlone'], label='anti-clones')

plt.xlabel('Time (steps)')
plt.ylabel('# clones/anti-clones')
plt.legend(loc='lower right')
plt.show()
