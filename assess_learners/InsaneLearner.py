import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose, self.learners = verbose, []
    def author(self):
        return "ytsang6"
    def add_evidence(self, train_x, train_y):
        for bag in range(20):
            bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
            bag_learner.add_evidence(train_x, train_y)
            self.learners.append(bag_learner)
    def query(self, points):
        results = []
        for learner in self.learners:
            results.append(learner.query(points))
        return np.mean(results, axis=0)
