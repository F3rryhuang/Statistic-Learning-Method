import numpy as np
class HiddenMarkov:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.forward_P = None
        self.backward_P = None
        self.stat_probs = None
        self.I = None
        self.best_P = None

    def backward_step(self, o, A, B, prev_beta):
        """compute beta[t]

        Args:
            o ([int]): observe at time-step t+1
            A ([matrix]): aij for prob that convert from status i at time-step t to status j at time-step t+1
            B ([matrix]): bj for prob that status j yields obseve at time-step t+1
            prev_beta ([matrix]): beta sequence at time-step t+1

        Returns:
            [matrix]: beta at time-step t
        """
        b = B[:, o]
        beta = A.dot(b * prev_beta)
        return beta

    def backward(self, O, A, B, pi):
        """ back_alg iterator

        Args:
            O ([list]): contains observed sequence
            A ([matrix]): aij for prob that convert from status i at time-step t to status j at time-step t+1
            B ([matrix]): bij for at time-step t, status i yields observe output j
            pi ([ndarray]): PIi for prob of status i serves as initial status

        Returns:
            [matrix]: all-time beta-matrix
            [float]: condition probability of O given A, B, PI
        """
        T = len(O)
        N = A.shape[0]
        beta = np.zeros((N, T))
        beta[:, -1] = 1             ## at time step T, beta[i] = 1
        for t in reversed(range(T - 1)):
            o = O[t + 1]
            beta[:, t] = self.backward_step(o, A, B, beta[:, t+1])
        
        self.beta = beta
        self.backward_P = float(pi.dot(B[:, O[0]] * beta[:, 0]))


    def forward_step(self, o, A, B, prev_alpha):
        """compute alpha[t + 1]

        Args:
            o ([int]): observe at time-step t+1
            A ([matrix]): aij for prob that convert from status i at time-step t to status j at time-step t+1
            B ([matrix]): bj for prob that status j yields obseve at time-step t+1
            prev_alpha ([matrix]): alpha sequence at time-step t+1

        Returns:
            [matrix]: alpha at time-step t+1
        """
        b = B[:, o]
        alpha = (A.T).dot(prev_alpha) * b
        return alpha


    def forward(self, O, A, B, pi):
        """ back_alg iterator

        Args:
            O ([list]): contains observed sequence, of shape (T)
            A ([matrix]): aij for prob that convert from status i at time-step t to status j at time-step t+1, of shape (N, N)
            B ([matrix]): bij for at time-step t, status i yields observe output j, of shape (N, n)
            pi ([ndarray]): PIi for prob of status i serves as initial status, of shape (1, N)

        Returns:
            [matrix]: all-time alpha-matrix
            [float]: condition probability of O given A, B, PI
        """
        T = len(O)
        N = A.shape[0]
        alpha = np.zeros((N, T))
        alpha[:, 0] = (pi.T * B[:, O[0]]).diagonal()        # (3, 1) * (3, 1) = (3, 3), elementwise products lay in diagonal
        for t in range(T - 1):
            o = O[t + 1]
            alpha[:, t + 1] = self.forward_step(o, A, B, alpha[:, t])
        prob = sum(alpha[:, -1])
        self.alpha =  alpha
        self.forward_P =  prob

    

    def status_prob(self, alpha, beta, *Args):
        """compute probability of status i at time-step t

        Args:
            alpha ([matrix]): alpha of all status of all time-steps
            beta ([matrix]): beta of all status of all time-steps
            i ([int]): index of status 
            t ([int]): index of time-step

        Returns:
            [float]: probability of status i at time-step t
        """
        if (Args):
            i, t = Args
            ## return prob at certain i and t
            self.stat_probs = alpha[i, t] * beta[i, t] / np.expand_dims(alpha[:, t], axis = 0).dot(np.expand_dims(beta[:, t], axis = 1))

        else:
            N = alpha.shape[0]
            stat_prob = np.zeros((N, N))
            stat_prob = np.multiply(alpha, beta)
            total_prob = np.sum(stat_prob, axis = 0)
            stat_prob /= total_prob
            ## return the whole
            self.stat_probs = stat_prob

    def Viterbi(self, O, A, B, pi):
        """Viterbi aims to find the best road leading to observe, given A, B, pi

        Args:
            O ([list]): contains observed sequence
            A ([matrix]): aij for prob that convert from status i at time-step t to status j at time-step t+1
            B ([matrix]): bij for at time-step t, status i yields observe output j
            pi ([ndarray]): PIi for prob of status i serves as initial status
        
        Returns:
            [float]: probability of the best road
            [ndarray]: index of node of the best road at each time-step
        """
        N = A.shape[0]
        T = len(O)
        Sigma = np.zeros((N, T))            ## Sigma(i, j) - the biggest probability of status i at t.s. t with o1, o2, ..., ot
        Psi = np.zeros((N, T))              ## Psi(i, j) - the node at t.s. t-1 of  the best road
        I = np.zeros(T)

        Sigma[:, 0] = pi * B[:, O[0]]
        Psi[:, 0] = 0
        
        for t in range(T - 1):
            Sigma[:, t + 1] = (np.max(Sigma[:, t].reshape(3, 1) * A, axis = 0) * B[:, O[t + 1]]).T
            Psi[:, t + 1] = np.argmax(Sigma[:, t].reshape(3, 1) * A, axis = 0).T
        best_P = np.max(Sigma[:, -1])
        I[-1] = np.argmax(Sigma[:, -1])

        for t in reversed(range(T - 1)):
            I[t] = Psi[int(I[t + 1]), t + 1]

        self.I = I
        self.best_P = best_P