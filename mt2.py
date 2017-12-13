# states => 0 = k , 1 = w
# observations => 0 = g , 1 = k, 2 = w

states = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
observations = [0, 0, 0, 2, 0, 0]
p_jump = 0.9  # jumping to next state prob
p_repeat = 0.1  # repeating same state prob
moveCount = len(observations)  # How many moves we will observe
currentMove = 1  # index of current move

# Calculating prior probabilities, probability being on state = X before observation (currentMove) = Y.
# Initially they are all zero.
w, h = len(states), moveCount + 1
prior_probabilities = [[0 for x in range(w)] for y in range(h)]


# Returns the probability of reaching current state from previous state using jump and repeat probabilities
def get_state_transition_probability(prevStateIndex, currentStateIndex):
    if prevStateIndex == (len(states) - 1) and currentStateIndex == 0:
        return p_jump
    elif prevStateIndex == currentStateIndex:
        return p_repeat
    elif currentStateIndex == 0:
        return 0
    elif prevStateIndex == (currentStateIndex - 1):
        return p_jump
    else:
        return 0


def calculate_repeat_move_coefficient(jumpCount, repeatCount):
    # tekrarli permutasyon
    return calculate_factorial(jumpCount + repeatCount) / (
        calculate_factorial(jumpCount) * calculate_factorial(repeatCount))


def calculate_factorial(num):
    if num == 1:
        return 1
    elif num == 0:
        return 1
    else:
        return num * (calculate_factorial(num - 1))


def get_state_observation_prob(state, observation):
    if state == 0 and observation == 0:
        return 0.7  # observing g when state is k
    if state == 0 and observation == 1:
        return 0.3  # observing k when state is k
    if state == 0 and observation == 2:
        return 0  # observing w when state is k
    if state == 1 and observation == 0:
        return 0.8  # observing g when state is w
    if state == 1 and observation == 1:
        return 0  # observing k when state is w
    if state == 1 and observation == 2:
        return 0.2  # observing w when state is w


def calculate_evidence_probability(currentMove, observation):
    prob = 0
    i = 0
    while i < len(states):
        tmp = prior_probabilities[currentMove - 1][i] * get_state_observation_prob(states[i], observation)
        prob = prob + tmp
        i = i + 1
    return prob


def calculate_state_probability_on_observation(state, observation, currentMove):
    # Applying bayes
    p_likelihood = get_state_observation_prob(states[state], observation)
    p_prior = calculate_prior_probability(currentMove, state)
    p_evidence = calculate_evidence_probability(currentMove, observation)
    if p_evidence == 0:
        return 0
    return p_likelihood * p_prior / p_evidence


def calculate_prior_probability(currentMove, state):
    prob = 0
    prevMove = currentMove - 1
    i = 0
    while i < len(states):
        prevStateProbability = prior_probabilities[prevMove][i]
        transitionProbability = get_state_transition_probability(i, state)
        prob = prob + (prevStateProbability * transitionProbability)
        i = i + 1
    return prob

    # mv = 1
    # st = 0
    # while mv <= moveCount:
    #     while st < len(states):
    #         if st > mv:
    #             prior_probabilities[mv][st] = 0
    #         elif st == mv:
    #             prior_probabilities[mv][st] = p_jump ** mv
    #         else:
    #             repeat = mv - st
    #             jump = mv - repeat
    #             prob = p_repeat ** repeat * p_jump ** jump * calculate_repeat_move_coefficient(repeat,
    #                                                                                            jump)  # more than one way
    #             #  to reach that state while jumping and repeating
    #             prior_probabilities[mv][st] = prob
    #         st = st + 1
    #     mv = mv + 1
    #     st = 0


def initialize_prior_probabilities():
    prior_probabilities[0][0] = 1
    prior_probabilities[0][1] = 0
    prior_probabilities[0][2] = 0


initialize_prior_probabilities()

currentMove = 1
st = 0
while currentMove <= len(observations):
    while st < len(states):
        posterior_prob = calculate_state_probability_on_observation(st, observations[currentMove - 1], currentMove)
        prior_probabilities[currentMove][st] = posterior_prob
        st = st + 1
    currentMove = currentMove + 1
    st = 0

# print results
print("After observations, probability distribution of the states are:")
st = 0
while st < len(states):
    print("Probability of state#", st, prior_probabilities[len(observations)][st])
    st = st + 1



# print(calculate_state_probability_on_observation(0, 0, 1))
# print(calculate_state_probability_on_observation(1, 0, 2))
# print(calculate_state_probability_on_observation(2, 0, 2))
