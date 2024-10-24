import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []  # for saving the learners

    def author(self):
        return "ytsang6"

    def add_evidence(self, train_x, train_y):
        bag = 0
        while bag < self.bags:
            if self.kwargs is not None and True:
                the_learner = self.learner(**self.kwargs)
            else:
                the_learner = self.learner()
            data_i = np.random.choice(range(train_x.shape[0]), size=train_x.shape[0], replace=True)
            the_train_x = train_x[data_i]
            the_train_y = train_y[data_i]
            the_learner.add_evidence(the_train_x, the_train_y)
            self.learners.append(the_learner)
            bag += 1

    def query(self, points):
        results = []
        for learner in self.learners:
            Y = learner.query(points)
            results.append(Y)
        return np.mean(results, axis=0)
