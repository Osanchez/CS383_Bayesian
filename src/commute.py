import numpy as np
import random as rand


class MarkovChainMonteCarlo:
    def __init__(self):
        pass

    def prior_sample(self, bn):
        """
        randomly selects samples for each joint probability based on parents
        :return: x - a prior sample
        """
        x = np.zeros(3)

        #  first joint prob
        random_choice = np.random.choice(bn[0], 1, bn[0].all(), bn[0])
        x[0] = random_choice[0]

        #  Second Joint Prob
        if x[0] == 0.1:
            random_choice = np.random.choice(bn[1][0], 1, bn[1][0].all(), bn[1][0])
            x[1] = random_choice
        elif x[0] == 0.9:
            random_choice = np.random.choice(bn[1][1], 1, bn[1][1].all(), bn[1][1])
            x[1] = random_choice

        #  Third Joint Prob
        if random_choice[0] == 0.8 or random_choice == 0.1:
            random_choice = np.random.choice(bn[2][0], 1, bn[2][0].all(), bn[2][0])
            x[2] = random_choice
        else:
            random_choice = np.random.choice(bn[2][1], 1, bn[2][1].all(), bn[2][1])
            x[2] = random_choice
        return x

    def gibbs_ask_traffic(self, X, e, Z, bn, N):
        """
        takes an initial state in the form {true or none, false or none,true or none} - {Rain, Traffic, Late}
        local variables:
            X, the current state of the network, initially copied from e
            e, observed values
            Z, the nonevidence variable in bn - in this case traffic being true or false
            bn, the bayesian network model
            N, a vector of counts for each value of X, initially zero
        :return: x - a prior sample with evidence variables
        """

        #makes copies
        X = e
        e = e

        #probability
        probability = [0,0]
        numerator = 0


        #True, False

        for x in range(N):
            #  second joint
            if Z == True: # if non evidence variable
                random_choice = np.random.choice([0,1], 1, True, [0.5, 0.5])[0] #Rain or No Rain
                X[1] = bn[1][random_choice][0]
            else:
                random_choice = np.random.choice([0, 1], 1, True, [0.5, 0.5])[0] #Rain or No Rain
                X[1] = bn[1][random_choice][1]

            #  first joint
            if X[1] == 0.8 or X[1] == 0.2: # Rain is true
                X[0] = bn[0][0]
            else:  # Rain is False
                X[0] = bn[0][1]

            #  third joint
            if X[1] == 0.8 or X[1] == 0.1: # traffic
                random_late = np.random.choice([0,1], 1, True, [0.5,0.5])[0]
                X[2] = bn[2][0][random_late]
            else:  # no traffic
                random_late = np.random.choice([0, 1], 1, True, [0.5, 0.5])[0]
                X[2] = bn[2][1][random_late]

            # print(X)
            if X[0] == 0.1:
                probability[0] += 1
            else:
                probability[1] += 1


        probability[0] = probability[0] / N
        probability[1] = probability[1] / N
        # print(probability)
        return probability


def main():
    p_r = np.array([0.1, 0.9])  # [P(r = true), P(r = false)]
    p_t_given_r = np.array([[0.8, 0.2], [0.1, 0.9]])  # [P(t|r = true), ¬P(t|r = true)], [P(t|r = false), ¬P(t|r = false)]
    p_l_given_t = np.array([[0.3, 0.7], [0.1, 0.9]])  # [P(l|t = true), ¬P(l|t = true)], [P(l|t = false), ¬P(l|t = false)]

    bn = [p_r, p_t_given_r, p_l_given_t]

    MCMC = MarkovChainMonteCarlo()
    p10 = MCMC.gibbs_ask_traffic([0, 0.1, 0], [0, 0.1, 0], True, bn, 10)
    p100 = MCMC.gibbs_ask_traffic([0, 0.1, 0], [0, 0.1, 0], True, bn, 100)
    p1000 = MCMC.gibbs_ask_traffic([0, 0.1, 0], [0, 0.1, 0], True, bn, 1000)

    """
    print("P(R|T) =  " + str(round(((0.8 * 0.1)/0.17), 3)))
    print("10 samples: " + str(p10[0]))
    print("100 samples: " + str(p100[0]))
    print("1000 samples: " + str(p1000[0]))
    """

    print(((0.8 * 0.1)/0.17))
    print(p10[0])
    print(p100[0])
    print(p1000[0])


if __name__ == '__main__':
    main()
