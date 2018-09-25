import Orange
from unions import CN2UnorderedLearner
import unions
from Orange.data import Table
import numpy as np
from scipy import stats
from Orange.evaluation.testing import sample


def class_helper(val,data):
    return data.domain.class_var.values.index(val)


def distance(predictedclasses,values,data):
    def distance_helper(val):
        return values.index(data.domain.class_var.values[val])
    templista = []
    for row in predictedclasses:
        temp = abs(distance_helper(row[0]) - distance_helper(row[1]))
        templista.append(temp)
    return np.mean(templista)


def desc_union_len(val, lista):
    return len(lista[:val])


def asc_union__len(val, lista):
    return len(lista[val:])


def conclind(rule,vallist):
    for val in vallist:
        if(val in str(rule)):
            return vallist.index(val)

    return -1


class Ocn2:

    def __init__(self, train_data,test_data,evaluator,beam_width,minex,maxlen,values,search_strategy):
        self.lista = []
        self.predictionsprob=[]
        self.predictions=[]
        self.klasy=[]
        self.train_data = train_data
        self.test_data = test_data
        self.values = values
        self.ruleset = []
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.minex = minex
        self.maxlen=maxlen
        self.search_strategy = search_strategy

    def createmodel(self):

        learner = CN2UnorderedLearner()
        learner.rule_finder.quality_evaluator = self.evaluator
        learner.rule_finder.search_algorithm = self.search_strategy
        learner.rule_finder.search_algorithm.beam_width = self.beam_width
        learner.rule_finder.search_strategy.constrain_continuous = True
        learner.rule_finder.search_strategy.union = True
        learner.rule_finder.search_strategy.rnd = False
        learner.rule_finder.general_validator.min_covered_examples = self.minex
        learner.rule_finder.general_validator.max_rule_length = self.maxlen

        learnerdesc = CN2UnorderedLearner()
        learnerdesc.rule_finder.quality_evaluator = self.evaluator
        learnerdesc.rule_finder.search_algorithm = self.search_strategy
        learnerdesc.rule_finder.search_algorithm.beam_width = self.beam_width
        learnerdesc.rule_finder.search_strategy.constrain_continuous = True
        learnerdesc.rule_finder.search_strategy.union = False
        learnerdesc.rule_finder.search_strategy.rnd = False
        learnerdesc.rule_finder.general_validator.min_covered_examples = self.minex
        learnerdesc.rule_finder.general_validator.max_rule_length = self.maxlen

        ruleset = []
        ruleset_desc = []

        for i, val in enumerate(self.values):
            temp_data = self.train_data.copy()
            temp_data_desc = self.train_data.copy()
            helper = self.values[self.values.index(val):]
            helper_desc = self.values[:self.values.index(val)]

            print(val)

            if i != 0:
                for d in temp_data:
                    if d.get_class() in helper:
                        d.set_class(val)
                learner.tar_klasa=class_helper(val, self.train_data)
                classifier = learner(temp_data)

                for rule in classifier.rule_list:
                    if val in str(rule):
                        ruleset.append(rule)

            if i != len(self.values) - 1:
                for d in temp_data_desc:
                    if d.get_class() in helper_desc:
                        d.set_class(val)

                learnerdesc.tar_klasa=class_helper(val,self.train_data)
                classifierdesc = learnerdesc(temp_data_desc)

                for rule in classifierdesc.rule_list:
                    if (val in str(rule)):
                        ruleset_desc.append(rule)

        finalruleset = []

        for rule in ruleset:
            finalruleset.append(rule)

        for rule in ruleset_desc:
            finalruleset.append(rule)

        self.ruleset=finalruleset

        def moje_predykcjie(X):

            num_classes = len(self.test_data.domain.class_var.values)
            probabilities = np.array([np.zeros(num_classes, dtype=float)
                                      for _ in range(X.shape[0])])

            num_hits = np.zeros(X.shape[0], dtype=float)
            total_weight = np.vstack(np.zeros(X.shape[0], dtype=float))
            for rule in finalruleset:
                if rule.length > 0:
                    curr_covered = rule.evaluate_data(X)
                    num_hits += curr_covered
                    temp = rule.curr_class_dist.sum()
                    probabilities[curr_covered] += rule.probabilities * temp
                    total_weight[curr_covered] += temp

            weigh_down = num_hits > 0

            probabilities[weigh_down] /= total_weight[weigh_down]
            return probabilities

        self.predictionsprob = moje_predykcjie(self.test_data.X)

        for i,row in enumerate(self.predictionsprob):
            temp = np.argmax(row)
            self.klasy.append([int(self.test_data.Y[i]), temp])
            self.predictions.append(temp)


class Ocn2r:

    def __init__(self, train_data,test_data,evaluator,beam_width,minex,maxlen,values,rat,search_strategy):
        self.lista = []
        self.predictionsprob=[]
        self.predictions=[]
        self.klasy=[]
        self.train_data = train_data
        self.test_data = test_data
        self.values = values
        self.rulset = []
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.minex = minex
        self.maxlen=maxlen
        self.rat = rat
        self.search_strategy = search_strategy

    def createmodel(self):

        learner = CN2UnorderedLearner()
        learner.rule_finder.quality_evaluator = self.evaluator
        learner.rule_finder.search_algorithm = self.search_strategy
        learner.rule_finder.search_algorithm.beam_width = self.beam_width
        learner.rule_finder.search_strategy.constrain_continuous = True
        learner.rule_finder.search_strategy.union = True
        learner.rule_finder.search_strategy.rnd = True
        learner.rule_finder.search_strategy.rat = self.rat
        learner.rule_finder.general_validator.min_covered_examples = self.minex
        learner.rule_finder.general_validator.max_rule_length = self.maxlen

        learnerdesc = CN2UnorderedLearner()
        learnerdesc.rule_finder.quality_evaluator = self.evaluator
        learnerdesc.rule_finder.search_algorithm = self.search_strategy
        learnerdesc.rule_finder.search_algorithm.beam_width = self.beam_width
        learnerdesc.rule_finder.search_strategy.constrain_continuous = True
        learnerdesc.rule_finder.search_strategy.union = False
        learnerdesc.rule_finder.search_strategy.rnd = True
        learnerdesc.rule_finder.search_strategy.rat = self.rat
        learnerdesc.rule_finder.general_validator.min_covered_examples = self.minex
        learnerdesc.rule_finder.general_validator.max_rule_length = self.maxlen

        ruleset = []
        ruleset_desc = []

        for i, val in enumerate(self.values):
            temp_data = self.train_data.copy()
            temp_data_desc = self.train_data.copy()
            helper = self.values[self.values.index(val):]
            helper_desc = self.values[:self.values.index(val)]


            if i != 0:
                for d in temp_data:
                    if d.get_class() in helper:
                        d.set_class(val)

                learner.tar_klasa=class_helper(val, self.train_data)
                classifier = learner(temp_data)

                for rule in classifier.rule_list:
                    if val in str(rule):
                        ruleset.append(rule)

            if i != len(self.values) - 1:
                for d in temp_data_desc:
                    if d.get_class() in helper_desc:
                        d.set_class(val)

                learnerdesc.tar_klasa=class_helper(val,self.train_data)
                classifierdesc = learnerdesc(temp_data_desc)

                for rule in classifierdesc.rule_list:
                    if val in str(rule):
                        ruleset_desc.append(rule)

        finalruleset = []

        for rule in ruleset:
            finalruleset.append(rule)

        for rule in ruleset_desc:
            finalruleset.append(rule)

        self.ruleset=finalruleset

        def moje_predykcjie(X):

            num_classes = len(self.test_data.domain.class_var.values)
            probabilities = np.array([np.zeros(num_classes, dtype=float)
                                      for _ in range(X.shape[0])])

            num_hits = np.zeros(X.shape[0], dtype=float)
            total_weight = np.vstack(np.zeros(X.shape[0], dtype=float))
            for rule in finalruleset:
                if rule.length > 0:
                    curr_covered = rule.evaluate_data(X)
                    num_hits += curr_covered
                    temp = rule.curr_class_dist.sum()
                    probabilities[curr_covered] += rule.probabilities * temp
                    total_weight[curr_covered] += temp

            weigh_down = num_hits > 0

            probabilities[weigh_down] /= total_weight[weigh_down]
            return probabilities

        self.predictionsprob = moje_predykcjie(self.test_data.X)

        for i,row in enumerate(self.predictionsprob):
            temp = np.argmax(row)
            self.klasy.append([int(self.test_data.Y[i]), temp])
            self.predictions.append(temp)


class Ocn2rn:

    def __init__(self, train_data,test_data,evaluator,beam_width,minex,maxlen,values,n,rat,search_strategy):
        self.train_data = train_data
        self.test_data = test_data
        self.values = values
        self.n = n;
        self.predictionlist = []
        self.predictions = []
        self.klasy = []
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.minex = minex
        self.maxlen=maxlen
        self.rat = rat
        self.search_strategy=search_strategy

    def createmodel(self):
        for i in range(self.n):
            learner = Ocn2r(self.train_data,self.test_data,self.evaluator,
                            self.beam_width,self.minex,self.maxlen,self.values,self.rat,self.search_strategy)
            learner.createmodel()
            self.predictionlist.append(learner.predictions)

    def defineclasses(self):
        temp = []
        for row in self.predictionlist[1]:
            sublist = []
            temp.append(sublist)

        for i,tab in enumerate(self.predictionlist):
            for i, value in enumerate(tab):
                temp[i].append(value)

        for i, row in enumerate(temp):
            stat=stats.mode(row)
            self.klasy.append([int(self.test_data.Y[i]), stat.mode[0]])
            self.predictions.append(stat.mode[0])


fileslist=['datasets/breast-cancer_nm.tab','datasets/breast-w_nm.tab','datasets/car.tab','datasets/cpu.tab','datasets/dataset1.tab','datasets/dataset3.tab','datasets/denbosch.tab','datasets/ERA.tab','datasets/ESL.tab','datasets/LEV.tab','datasets/SWD.tab','datasets/windsor.tab']
classeslist=['datasets/breast-cancer_nm_classes.txt','datasets/breast-w_nm_classes.txt','datasets/car_classes.txt','datasets/cpu_classes.txt','datasets/dataset1_classes.txt','datasets/dataset3_classes.txt','datasets/denbosch_classes.txt','datasets/ERA_classes.txt','datasets/ESL_classes.txt','datasets/LEV_classes.txt','datasets/SWD_classes.txt','datasets/windsor_classes.txt']


searchstrats=[unions.BeamSearchAlgorithm(),unions.RandomBeamSearchAlgorithm,unions.WeightedRandomSearchAlgorithm()]
evals = [unions.EntropyEvaluator(), unions.LaplaceAccuracyEvaluator(), unions.WeightedRelativeAccuracyEvaluator(),unions.TruePositiveRateEvaluator()]
beam_width = 20
minex = 25
maxlen = 8
rat = 0.5

data = Table('datasets/car.tab')
text_file = open('datasets/car_classes.txt', "r")
values = text_file.read().split(',')
train_data, test_data = sample(data, n=0.7, stratified=True, replace=False, random_state=0)

#ocn2test = Ocn2(train_data, test_data,evaluator,beam_width,minex,maxlen,values,searchstrats[0])
ocn2test = Ocn2rn(train_data, test_data,evals[0],beam_width,minex,maxlen,values,5,0.5,searchstrats[0])

ocn2test.createmodel()

ocn2test.defineclasses()
print(distance(ocn2test.klasy,values,train_data))
