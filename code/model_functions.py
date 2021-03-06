#Functions used for bvm_model, ba_model

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

def getPercentPolarized(model):
    #returns the percentage of the populace that is either a clone or anticlone
    num = getNumClonePairs(model) + getNumAnticlonePairs(model)
    den = scipy.special.binom(model.num_agents,2)
    return num/den

def getNumClonePairs(model):
    # Return the number of pairs of agents who "agree" (opinion in the same
    # cluster) on every issue.
    return getNumAgentPairsWithKAgreements(model,
            model.num_issues)

def getNumAnticlonePairs(model):
    # Return the number of pairs of agents who "disagree" (opinion in different
    # clusters) on every issue.
    return getNumAgentPairsWithKAgreements(model,0)

# BEGIN ELEGANT
#def getNumAgentPairsWithKAgreementsClosure(k):
#    # Return a function, which takes only a model as an argument, which will
#    # compute the number of pairs of agents who "agree" (opinion in the same
#    # cluster) on exactly k issues.
#    return lambda model: getNumAgentPairsWithKAgreements(model, k)
# END ELEGANT

# Since Mesa won't allow a lambda expression, the ridiculously bad code below
# is used instead of the elegant code above.  - SD
# BEGIN UGLY
def getNumAgentPairsWith1Agreements(model):
    pwa = getNumPairwiseAgreements(model)
    return sum([ a == 1 for a in pwa ])

def getNumAgentPairsWith2Agreements(model):
    pwa = getNumPairwiseAgreements(model)
    return sum([ a == 2 for a in pwa ])

def getNumAgentPairsWith3Agreements(model):
    pwa = getNumPairwiseAgreements(model)
    return sum([ a == 3 for a in pwa ])
# END UGLY

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
                printAllAgentOpinions(model)
                return 0
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

def validateBuckets(model):
    for a in model.schedule.agents:
        seen = False
        for buckKey, buckVal in model.buckets.items():
            if(a in buckVal):
                if(seen == True):
                    print(f"{a} in more than one bucket")
                    break
                else:
                    seen = True

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

def plotAgreementCensus(model):
        df = model.datacollector.get_model_vars_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['numClonePairs'], label='clones')
        ax.plot(df['numAnticlonePairs'], label='anti-clones')
        colors = get_cmap('Greys')
        for numAgreements in range(1,model.num_issues):
            ax.plot(df[f'num{numAgreements}AgreementPairs'],
                    label=f'{numAgreements} agreements' if numAgreements > 1 else
                    '1 agreement', color=colors(numAgreements/model.num_issues))

            ax.set_xlabel('Time (steps)')
        ax.set_ylabel('# clones/anti-clones/1 Agreements/2 Agreements')
        ax.legend(loc='best')

        ax2 = ax.twinx()
        ax2.plot(df['Buckets'], label='Buckets', color='maroon',
                linestyle="dashed")
        ax2.set_ylabel("Number of Buckets", color='maroon')
        ax2.axhline(y=0, linestyle = 'dotted')
        plt.annotate("Buckets: {}".format(len(model.buckets)), xy=(.9*model.steps,len(model.buckets)), fontsize='medium', fontweight='heavy', fontvariant='small-caps', fontstyle='italic')
        plt.title("WS: Census Plot: Density 0.10")
        plt.show()

def plotPolarizationMeasures(model):
        df = model.datacollector.get_model_vars_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['PercentPolarized'], label="%Polarized", color='steelblue')
        ax.set_title("WS: Measures of Issue Alignment: Density 0.10")
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('% polarized (# of clones+anticlones/ N choose 2)', color='steelblue')
        ax.set_yticks(np.arange(0,1.05,.1))
        ax.legend(loc='best')

        ax2 = ax.twinx()
        ax2.plot(df['Buckets'], label='Buckets', color='maroon',
                linestyle="dashed")
        ax2.set_ylabel("Number of Buckets", color='maroon')
        ax2.axhline(y=0, linestyle = 'dotted')
        plt.annotate("Buckets: {}".format(len(model.buckets)), xy=(.9*model.steps,len(model.buckets)), fontsize='medium', fontweight='heavy', fontvariant='small-caps', fontstyle='italic')
        plt.show()
def getDensity(model):
    return nx.density(model.G)
