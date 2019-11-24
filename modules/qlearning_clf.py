from . import util
from random import shuffle


class featureExtractor():
    """Object for extracting features from an image file. The data is provided in a dataframe variable
    with a label_names column and an index column for state name.
    """

    def __init__(self, df):
        self.states = list(df['index'])
        self.features = list(df['hog_features'])

    def getFeatures(self, state, action):
        return self.features[state]


class QLearningClassifier:
    """Agent to use for project. Should be a modified version of ApproximateQAgent from Project/Homework 4
    in CS557.
    """

    def __init__(self, df, epsilon=0.05, gamma=0.8, alpha=0.2):
        # feature extractor object has a method getFeatures(..) that provides a set of features for any given
        # state (i.e. image). It takes an action but this does not affect the featurs returned
        self.featExtractor = featureExtractor(df)

        # alpha    - learning rate
        # epsilon  - exploration rate (Not sure what this is, this the random action factor?!?)
        # gamma    - discount factor
        # numTraining - number of training episodes, i.e. no learning after these many episodes
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.weights = util.Counter()
        self.labels = list(df['label_name'])
        self.legalActions = list(set(self.labels))
        self.discount = float(gamma)

    def getWeights(self):
        return self.weights

    def getLabel(self, state):
        return self.labels[state]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        Q_Values = util.Counter()
        actions = self.legalActions
        for a in actions:
            Q_Values[a] = self.getQValue(state, a)

        # Best action (maximizes Q-Value)
        max_action = Q_Values.argMax()
        return max_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        action = self.computeActionFromQValues(state)
        return action

    def getQValue(self, state, action):
        """For a given state, action pair it should return the dot product of the weight vector and the
        feature vector for that state. In our case the feature vector is the image descriptors for that image
        (note that an image is a state in our project).
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Q(state, action) = w dot featureVector
        features = self.featExtractor.getFeatures(state, action)
        QValue = sum(self.weights[i] * features[i] for i in features.keys())
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Q-Values for each a' :
        Q_Counter = util.Counter()

        # legal actions is trivial in our case as there are always as many actions as classes
        a_prime_values = self.legalActions
        for a_prime in a_prime_values:
            # Q-Value for a':
            Q_Counter[a_prime] = self.getQValue(nextState, a_prime)

        # difference = (R + gamma * max[Q(s',a')] ) - Q(s,a)
        difference = (reward + self.discount * Q_Counter[Q_Counter.argMax()]) - self.getQValue(state, action)

        # wi = wi + alpha * difference * fi(s,a)
        features = self.featExtractor.getFeatures(state, action)
        for i in features.keys():
            self.weights[i] = self.weights[i] + self.alpha * difference * features[i]
        return 0

    def train(self, epochs=10):
        """Run through the training using the trainign dataset.
        """

        for i, e in enumerate(range(epochs)):
            print('Epoch {} of {}'.format(i + 1, epochs))
            state_list = list(range(len(self.labels)))
            shuffle(state_list)

            # get the initial state
            state = state_list.pop()
            while len(state_list) > 0:
                # get the best action for this state by Q-values
                action = self.getAction(state)

                nextState = state_list.pop()

                # check if action matches label, if true then reward is 1 else it is 0

                if action == self.labels[state]:
                    reward = 10
                else:
                    reward = -10

                self.update(state, action, nextState, reward)

    def test(self, test_data):
        """Test method is given the test dataframe (data) and gives the best action for each image/state
        using trained weights of the class instance. The testing accuracy is reported."""
        legalActions = self.legalActions

        # report testing accuracy
        featExtractor = featureExtractor(test_data)
        labels = list(test_data['label_name'])

        correct_count = 0
        actions = self.legalActions

        state_list = list(range(len(labels)))

        for state in state_list:
            Q_Values = util.Counter()
            for a in actions:
                features = featExtractor.getFeatures(state, a)
                QValue = sum(self.weights[i] * features[i] for i in features.keys())
                Q_Values[a] = QValue

            # Best action (maximizes Q-Value)
            max_action = Q_Values.argMax()

            if max_action == labels[state]:
                correct_count += 1

        accuracy = float(correct_count) / float(len(state_list))
        return accuracy
