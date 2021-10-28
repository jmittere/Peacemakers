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
import matplotlib.colors as clr
from matplotlib.cm import get_cmap
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from graspologic.cluster.autogmm import AutoGMMCluster
import ipdb

from agent_bvm import bvmAgent


# The minimum difference in the opinion values of two agents in order for those
# opinions to be considered in two different clusters.
CLUSTER_THRESHOLD = .05
BUCKET_THRESHOLD = .05

# The number of consecutive iterations without any persuasions/replusions that
# the model will continue to run before stopping.
EQUILIBRIUM_THRESHOLD = 5

# The value of an opinion an agent would need to have to be considered "high"
# (and 1-this would be considered "low").
EXTREME_THRESHOLD = .9

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

def getMultimodalityStatistic(model):
    # Return a statistic estimating the evidence for multi-modality in the
    # number of opinion agreements that each agent has pairwise with every
    # other.
    #pwa = getNumPairwiseAgreements(model)

    # For now, just use a blunt object: number of anti-clones and clones.
    return (getNumAgentPairsWithKAgreements(model, 0) +
        getNumAgentPairsWithKAgreements(model, model.num_issues))

def getNumClonePairs(model):
    # Return the number of pairs of agents who "agree" (opinion in the same
    # cluster) on every issue.
    return getNumAgentPairsWithKAgreements(model,
        len(model.schedule.agents[0].opinions))

def getNumAnticlonePairs(model):
    # Return the number of pairs of agents who "disagree" (opinion in different
    # clusters) on every issue.
    return getNumAgentPairsWithKAgreements(model,0)

def getNumAgentPairsWithKAgreementsClosure(k):
    # Return a function, which takes only a model as an argument, which will
    # compute the number of pairs of agents who "agree" (opinion in the same
    # cluster) on exactly k issues.
    return lambda model: getNumAgentPairsWithKAgreements(model, k)

def getNumAgentPairsWithKAgreements(model, k):
    # Return the number of pairs of agents who "agree" (opinion in the same
    # cluster) on exactly k issues.
    pwa = getNumPairwiseAgreements(model)
    return sum([ a == k for a in pwa ])


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
                assort = nx.numeric_assortativity_coefficient(model.G,
                    "iss_{}".format(i))
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
    if (model.steps%2) != 0:
        return None

    oList = getAllOpinions(model,issueNum) #get all opinions for an issue
    opinion_arr = np.array(oList)
    opinion_arr = opinion_arr.reshape(model.num_agents,1)
    try:
        clusters = model.autogmm.fit(opinion_arr) #perform AutoGMM algorithm
    except RuntimeWarning as rw:
        print("Opinions for Issue {} in step {}: {}".format(issueNum,model.steps,opinion_arr))
        print("RuntimeWarning: ", rw)
        return None


    #print("There were {} clusters for issue {} in step {} ".format(clusters.n_components, issueNum, model.steps))
    return clusters.n_components_

def returnPersuasionsPerCapita(model):
    return model.persuasions / model.num_agents

def returnNumLowOpinions(model, issueNum):
    #returns the number of agents within EXTREME_THRESHOLD of 0
    agents = model.schedule.agents
    agentCount=0
    for a in agents:
        if a.opinions[issueNum]<=1-EXTREME_THRESHOLD:
            agentCount+=1
    return agentCount

def returnNumHighOpinions(model, issueNum):
    #returns the number of agents within EXTREME_THRESHOLD of 1
    agents = model.schedule.agents
    agentCount=0
    for a in agents:
        if a.opinions[issueNum]>=EXTREME_THRESHOLD:
            agentCount+=1
    return agentCount

def returnNumModerateOpinions(model, issueNum):
    #returns number of agents within (1-EXTREME_THRESHOLD,EXTREME_THRESHOLD)
    agents = model.schedule.agents
    agentCount=0
    for a in agents:
        if 1-EXTREME_THRESHOLD<=a.opinions[issueNum]<=EXTREME_THRESHOLD:
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

def updateBuckets(model):
    for a in model.schedule.agents:
        #print(f"{a}\t{model.buckets}")
        if not model.buckets: #if buckets dict is empty
            opinionKey = tuple(a.opinions)
            model.buckets[opinionKey] = [a]
        else: #buckets dict isn't empty
            for bucket in model.buckets.items():
                identical = True
                opinionVals = bucket[0] #key of buckets dict
                for i in range(0,model.num_issues):
                    bucketOpinion = opinionVals[i]
                    opinion = a.opinions[i]
                    if abs(opinion-bucketOpinion)>BUCKET_THRESHOLD: #agent isn't identical to this bucket's opinions
                        identical = False
                        break

                if identical: #add agent to this bucket
                    newKey = ()
                    for i in range(model.num_issues):
                        opinionAvg = ((opinionVals[i]*len(bucket[1])) + a.opinions[i])/(len(bucket[1])+1)
                        newKey += (opinionAvg,)

                    newVal = bucket[1]+[a]
                    model.buckets[newKey] = newVal #set new bucket
                    if newKey != opinionVals: #new key and old key arent the same, if they are the same, don't delete it
                        del model.buckets[opinionVals] #delete old bucket
                
                    #print(model.buckets[newKey])
                    break
            
            if not identical: #agent doesn't belong to any existing buckets, so create a new bucket for this agent
                #print("Not identical...creating new")
                opinionKey = tuple(a.opinions)
                model.buckets[opinionKey] = [a]
    
    return len(model.buckets)

def plotBuckets(model):
   fig = plt.figure()
   xlabels = []
   for j in model.buckets.keys():
       roundedKey = ""
       for element in j:
           roundedKey += str(round(element,2)) + ", "
       roundedKey = roundedKey[:-2] #remove comma and space
       xlabels.append(roundedKey)
   buckets = [len(i) for i in model.buckets.values()]
   plt.bar(xlabels, buckets, width=0.4)
   plt.title('Opinion Buckets')
   plt.ylabel('Number of Agents')
   plt.xlabel('Opinion Values for the Bucket')
   plt.show()
   

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

    # l_steps: max # of simulation iterations 
    # n_agents: # of agents
    # p: prob of edge for Erdos-renyi
    # issues: # of issues for each agent
    # o: openness threshold for agents
    # d: disgust threshold for agents
    def __init__(self, l_steps, n_agents, p, issues, o, d, seed=None):
        super().__init__()
        self.l_steps = l_steps
        self.num_agents = n_agents
        self.num_issues = issues
        self.openness_threshold = o
        self.disgust_threshold = d
        self.schedule = RandomActivation(self)
        self.steps = 0
        self.repulsions = 0
        self.persuasions = 0
        self.influencesLastStep = 0
        self.equilibriumCounter= 0
        self.running = True
        self.clusterTracking = {} #key:(unique_id, issue) 
        self.buckets = {} #key: tuple of mean opinions for the bucket, value: list of agents in that bucket

        self.autogmm = AutoGMMCluster(affinity='euclidean', 
                linkage='ward', covariance_type='full')

        # generate ER graph with n_agents nodes and prob of edge of p
        self.G = nx.erdos_renyi_graph(n_agents, p)
        while not nx.is_connected(self.G):
            self.G = nx.erdos_renyi_graph(n_agents, p)

        # instantiate and add agents
        for i in range(self.num_agents):
            agent = bvmAgent(i, self)
            self.G.nodes[i]["agent"] = agent
            self.schedule.add(agent)

        # create all the mesa "reporters" to gather stats
        reporters =  {"clustersforIssue_{}".format(i):
                lambda model, issueNum=i:
                getNumClusters(model,issueNum) for i in range(self.num_issues)}

        #autoGmmReporters = {"autogmmclustersforIssue_{}".format(i):lambda model, issueNum=i: doAutoGMM(model,issueNum) for i in range(self.num_issues)}

        repubs = {"low_iss_{}".format(i):
            lambda model, issueNum=i: returnNumLowOpinions(model,issueNum)
                for i in range(self.num_issues)}
        dems = {"high_iss_{}".format(i):
            lambda model, issueNum=i:returnNumHighOpinions(model,issueNum)
                for i in range(self.num_issues)}
        mods = {"mod_iss_{}".format(i):
            lambda model, issueNum=i: returnNumModerateOpinions(model,issueNum)
                for i in range(self.num_issues)}
        reporters.update(repubs)
        reporters.update(dems)
        reporters.update(mods)
        #reporters.update(autoGmmReporters)

        self.datacollector = DataCollector(
                model_reporters=reporters,
                agent_reporters={}
                )

        self.datacollector._new_model_reporter("Steps", getSteps)
        self.datacollector._new_model_reporter("Buckets", updateBuckets)
        #self.datacollector._new_model_reporter("assortativity", get_avg_assort)
        #self.datacollector._new_model_reporter("numberOfNonUniformIssues",
        #    numNonUniformIssues)
        self.datacollector._new_model_reporter("persuasions", getPersuasions)
        self.datacollector._new_model_reporter("repulsions", getRepulsions)

        self.datacollector._new_model_reporter("numClonePairs",
            getNumClonePairs)
        self.datacollector._new_model_reporter("numAnticlonePairs",
            getNumAnticlonePairs)

        for numAgreements in range(1,self.num_issues):
            self.datacollector._new_model_reporter(
                f"num{numAgreements}AgreementPairs",
                getNumAgentPairsWithKAgreementsClosure(numAgreements))

        self.datacollector.collect(self)


    def step(self):
        self.influencesLastStep = 0
        self.schedule.step()
        self.buckets = {} #TODO: fix buckets to persist through steps
        '''If this line is removed, the buckets do not get overwritten and there will be too many buckets due to the different bucket labels'''

        if self.influencesLastStep == 0:
            self.equilibriumCounter += 1
        else:
            # Reset equilibrium counter if there are persuasions or repulsions
            # still happening.
            self.equilibriumCounter = 0

        self.datacollector.collect(self)

        # Stop if exceeds step limit
        if self.l_steps == self.steps + 1:
            self.running = False

        # Stop if there have been no persuasions/replusion for longer than
        # threshold.
        if self.equilibriumCounter > EQUILIBRIUM_THRESHOLD:
            self.running = False

        self.steps += 1

if __name__ == "__main__":

    #lsteps, agents, p, issues, othresh, dthresh
    test = bvmModel(1000, 100, 0.3, 3, 0.1, 0.55)

    for i in range(test.l_steps):
        test.step()
        if(test.running == False):
            break
    
    #printAllAgentOpinions(test)
    df = test.datacollector.get_model_vars_dataframe()
    df.to_csv("singleRun.csv")
    print(df)
    #updateBuckets(test)
    print("Buckets: ", test.buckets)
    print("Number of buckets: ", len(test.buckets)) 

    for b,cnt in test.buckets.items():
        print(f"{len(cnt)} agents in Bucket: {b}")
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
    plt.plot(df['numClonePairs'], label='clones')
    plt.plot(df['numAnticlonePairs'], label='anti-clones')
    colors = get_cmap('Greys')
    for numAgreements in range(1,test.num_issues):
        plt.plot(df[f'num{numAgreements}AgreementPairs'],
            label=f'{numAgreements} agreements' if numAgreements > 1 else
                '1 agreement', color=colors(numAgreements/test.num_issues))

    plt.xlabel('Time (steps)')
    plt.ylabel('# clones/anti-clones/1 Agreements/2 Agreements')
    plt.legend(loc='best')
    
    ax2 = ax.twinx()
    ax2.plot(df['Buckets'], label='Buckets', color='maroon')
    ax2.set_ylabel("Number of Buckets", color='maroon')
    plt.show()
