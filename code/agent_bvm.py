#agent_bvm.py

from mesa import Agent
import networkx as nx

MIN_OPINION_MOVEMENT_FOR_PERSUASION = .03

class bvmAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinions = []
        for i in range(int(self.model.num_issues)):
            num = round(self.random.uniform(0, 1), 2)
            self.opinions.append(num)
        self.pushOpinionsToGraph()

    def pushOpinionsToGraph(self):
        # Sets a node attribute in network x graph for each issue.
        issues = {"iss_{}".format(i): round((self.opinions[i]*100))
            for i in range(self.model.num_issues)}
        for key in issues:
            self.model.G.nodes[self.unique_id][key] = issues[key]

    def step(self):
        self.takeInfluence()
        self.pushOpinionsToGraph()

    def takeInfluence(self):
        neighbors = list(self.model.G.neighbors(self.unique_id))
        influencer = self.model.random.choice(neighbors)

        # CI2 -- Select two (distinct) issues at random: one for comparison,
        # and one for persuasion/repulsion if the comparison difference is
        # close/far enough.
        persuade_index = self.random.randint(0, int(self.model.num_issues)-1)
        compare_index = self.random.randint(0, int(self.model.num_issues)-1)
        while(persuade_index == compare_index):
            compare_index = self.random.randint(0, self.model.num_issues-1)

        my_compare_val = self.opinions[compare_index]
        my_persuade_val = self.opinions[persuade_index]
        influencer_compare_val = (self.model.G.nodes[influencer]["agent"].
            opinions[compare_index])
        influencer_persuade_val = (self.model.G.nodes[influencer]["agent"].
            opinions[persuade_index])

        # Compares this agent's opinion to its influencer to see whether it
        # can be attracted.
        if (abs(my_compare_val-influencer_compare_val) <=
            self.model.openness_threshold):

            # Yes, the two agents are similar enough on their comparison issue
            # to warrant this agent being attracted to the influencer on the
            # persuasion issue. Split the difference.
            self.opinions[persuade_index] = round(
                (my_persuade_val + influencer_persuade_val)/2,2)
            if (abs(my_persuade_val - self.opinions[persuade_index]) <
                MIN_OPINION_MOVEMENT_FOR_PERSUASION):
                #doesn't count as persuasion
                pass
            else:
                self.model.persuasions+=1

        elif (abs(my_compare_val-influencer_compare_val) <=
            self.model.disgust_threshold):

            # The two agents are actually far apart enough on their comparison
            # issue to warrant this node being repelled from the influencer on
            # the persuasion issue. Move this agent's persuasion opinion away 
            # from the influencer's by an amount equal to the amount it would
            # have moved *towards* the influencer's if it had been an
            # attraction.
            attractionOpinion = round((my_persuade_val +
                influencer_persuade_val)/2,2)
            repulsionAmt = abs(attractionOpinion-my_persuade_val)

            self.model.repulsions+=1
            if my_compare_val>influencer_compare_val:
                # The agent's opinion is higher than the influencer's, so it
                # should be pushed closer to 1.
                self.opinions[persuade_index]+=repulsionAmt
                if(self.opinions[persuade_index]>=1):
                    self.opinions[persuade_index] = .99
            else:
                # The agent's opinion is lower than the influencer's, so it
                # should be pushed closer to 0.
                self.opinions[persuade_index]-=repulsionAmt
                if(self.opinions[persuade_index]<=0):
                    self.opinions[persuade_index] = .01

        else: #no attraction or repulsion
            pass

