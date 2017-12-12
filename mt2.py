import numpy as np

# states => 0 = k , 1 = w
# observations => 0 = g , 1 = k, 2 = w

states = [0, 0, 1]
observations = [0, 0]
p_jump = 0.9  # jumping to next state prob
p_repeat = 0.1  # repeating same state prob
moveCount = len(observations)  # How many moves we will observe
currentMove = 1  # index of current move

# Calculating prior probabilities, probability being on state = X before observation (currentMove) = Y.
# Initially they are all zero.
w, h = len(states), moveCount + 1
prior_probabilities = [[0 for x in range(w)] for y in range(h)]


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


def calculate_evidence_probability(state, currentMove, observation):
    prob = 0
    i = 0
    while i < len(states):
        tmp = prior_probabilities[currentMove][state] * get_state_observation_prob(states[state], observation)
        prob = prob + tmp
        i = i + 1
    return prob


def calculate_state_probability_on_observation(state, observation, currentMove):
    # Applying bayes
    p_likelihood = get_state_observation_prob(states[state], observation)
    p_prior = calculate_prior_probability(state, currentMove)
    p_evidence = calculate_evidence_probability(state, currentMove, observation)
    return p_likelihood * p_prior / p_evidence


def calculate_prior_probability(state, currentMove):
    # Applying recursive bayes
    if currentMove == 1:
        # At first step, it can be on state_0 or state_1, setting further step probabilities to zero.
        i = 2
        while i < len(states):
            prior_probabilities[1][i] = 0
            i = i + 1
        prior_probabilities[1][
            0] = p_repeat  # being on the initial state probability on the move=1 is equal to repeat probability
        prior_probabilities[1][1] = p_jump  # being on state 1 is equal to jump probability
        return prior_probabilities[currentMove][state]
    else:
        # Highest possible state we can get at this move is State(move) assuming we never repeat the same state.
        # Therefore setting higher state probabilities as zero
        i = state + 1
        while i < len(states):
            prior_probabilities[1][i] = 0
            i = i + 1
        if state > currentMove:
            prior_probabilities[currentMove][state] = 0
            return prior_probabilities[currentMove][state]
        elif state == currentMove:
            prob = (state + 1) ** p_jump
            prior_probabilities[currentMove][state] = prob
            return prior_probabilities[currentMove][state]
        else:
            repeat = currentMove - state
            jump = currentMove - repeat
            prob = p_repeat ** repeat * p_jump ** jump * calculate_repeat_move_coefficient(repeat,
                                                                                           jump)  # more than one way
            #  to reach that state while jumping and repeating
            prior_probabilities[currentMove][state] = prob
            return prior_probabilities[currentMove][state]


# print(calculate_state_probability_on_observation(1, 0, 1))
print(calculate_state_probability_on_observation(1, 0, 2))
# print(calculate_state_probability_on_observation(2, 0, 2))
