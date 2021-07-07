from HMM_utils import *

HMM = HiddenMarkov()
A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])

pi = np.array([[0.2, 0.4, 0.4]])
O = [0, 1, 0, 1]

HMM.backward(O, A, B, pi)
HMM.forward(O, A, B, pi)
HMM.status_prob(HMM.alpha, HMM.beta)
HMM.Viterbi(O, A, B, pi)

print(HMM.backward_P)
print(HMM.forward_P)
print(HMM.stat_probs)
print(HMM.I)
print(HMM.best_P)