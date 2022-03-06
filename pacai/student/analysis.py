"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Set noise to 0
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    Decrased discount, but left it positive, decreased noise to be negative,
    and living reward is also negative.
    """

    answerDiscount = 0.1
    answerNoise = -0.2
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Decrased discount and negated living reward.
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -0.9

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Set both noise and living reward to negative.
    """

    answerDiscount = 0.6
    answerNoise = -0.2
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Set living reward to negative.
    """

    answerDiscount = 0.8
    answerNoise = 0.2
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Increased reward to avoid suicide. Although also decrased discount.
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 100

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    The desired effect doesnt seem to be possible.
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
