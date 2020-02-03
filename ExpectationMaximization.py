import random
import maths


class EM_Algorithm:
    def __init__(self, data=None, num_classes=None):
        self.data = data
        self.num_classes = num_classes
        self.class_counts = None
        self.feature_counts = None

    # returns a distribution over the classes for the original tuple in the current model
    def prob(self, tple, class_counts, feature_counts):
        features = self.data
        unnorm = [prod(feature_counts[i][feat(tple)][c]
                       for (i, feat) in enumerate(features))/(class_counts[c]**(len(features)-1))
                    for c in range(self.num_classes)]
        return [prob/sum(unnorm) for prob in unnorm]

    # This function will update the model.
    def em_step(self, orig_class_counts, orig_feature_counts):
        class_counts = [0]*self.num_classes
        feature_counts = [{val: [0]*self.num_classes for val in self.data}]
        for tple in self.data:
            if orig_class_counts:
                # with no model we will have a random distribution.
                tple_class_dist = self.prob(tple=tple, class_counts=orig_class_counts, feature_counts=orig_feature_counts)
            else:
                tple_class_dist = random_dist(self.num_classes)
            for class_ in range(self.num_classes):
                class_counts[class_] += tple_class_dist[class_]
                for (index, feature) in enumerate(self.data):
                    feature_counts[index][feature(tple)][class_] += tple_class_dist[class_]
        return class_counts, feature_counts

    # we could add a stopping criteria with a tolerance instead of a predifined number of steps.
    def fit(self, n_steps):
        for step in range(n_steps):
            self.class_counts, self.feature_counts = self.em_step(orig_class_counts=self.class_counts,
                                                                  orig_feature_counts=self.feature_counts)
            
    def log_loss(self, tple):
        features = self.data
        result = 0
        count_class = self.class_counts
        count_feature = self.feature_counts
        for count in range(self.num_classes):
            result += prod(count_feature[i][feature(tple)][count]
                           for (i, feature) in enumerate(features))/(count_class[count]**len(features)-1)
        if result > 0:
            return -(math.log2(result/len(self.data)))                                   
        else:
            return float("inf")
