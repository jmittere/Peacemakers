#agent_bvm.py

from mesa import Agent
import networkx as nx

class bvmAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.opinions = []
        self.initialOpinions = []
        for i in range(int(self.model.num_issues)):
            num = round(self.random.uniform(0, 1), 2)
            self.opinions.append(num)
            self.initialOpinions.append(num)

    def updateOpinions(self):
        #sets the node attribute in network x graph for each issue
        issues = {"iss_{}".format(i): round((self.opinions[i]*100)) for i in range(self.model.num_issues)}
        for key in issues:
            self.model.G.nodes[self.unique_id][key] = issues[key]

    def step(self):     
        self.takeInfluence()
        self.updateOpinions()

    def takeInfluence(self):
        neighbors = list(self.model.G.neighbors(self.unique_id))
        influencer = self.model.random.choice(neighbors)
        persuade_index = self.random.randint(0, int(self.model.num_issues)-1)
        compare_index = self.random.randint(0, int(self.model.num_issues)-1)

        while(persuade_index == compare_index):
            compare_index = self.random.randint(0, self.model.num_issues-1)

        node_compare = self.opinions[compare_index]
        node_persuade = self.opinions[persuade_index]
        influencer_compare = self.model.G.nodes[influencer]["agent"].opinions[compare_index]
        influencer_persuade = self.model.G.nodes[influencer]["agent"].opinions[persuade_index]
        
        #compares node and influencer opinions to see if node can be influenced 
        if(abs(node_compare-influencer_compare) <= self.model.compare_threshold):
            #node can be influenced, so set their new opinion as the average of their old opinion and influencer's opinion
            self.opinions[persuade_index] = round((node_persuade + influencer_persuade)/2,2)
            if(abs(node_persuade - self.opinions[persuade_index]) < .03):
                #doesn't count as persuasion
                pass
            else:
                self.model.persuasions+=1
