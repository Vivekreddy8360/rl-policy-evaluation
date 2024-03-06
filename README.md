# POLICY EVALUATION

## AIM
To develope the python program to evaluate the policy.
## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: *G: The goal state & **H*: A hole state.
* Five Transition states / Non-terminal States including  *S*: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* *50%* chance that the agent moves in the intended direction.
* *33.33%* chance that the agent stays in its current state.
* *16.66%* chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
<p align="center">
<img width="600" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>

## POLICY EVALUATION FUNCTION
### Formula
<img width="350" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

## POLICY EVALUATION FUNCTION
```
Developed by: M.vivek reddy
Reg No: 212221240030
```
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
   	'''Initialize 1st Iteration estimates of state-value function(V) to zero'''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        '''Initialize the current iteration estimates to zero'''
        V=np.zeros(len(P),dtype=np.float64)
        
        for s in range(len(P)):
        
            '''Update the value function for each state'''
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                
            '''Check for convergence'''
            if np.max(np.abs(prev_V-V))<theta:
                break
                
            '''Update the previous state-value function'''
            prev_V=V.copy()
        return V
```

## OUTPUT:
## policy 1
```
Policy:
|           | 01      < | 02      < | 03      < | 04      < | 05      < |           |
State-value function:
|           | 01 0.00246 | 02   0.01 | 03 0.03322 | 04 0.10489 | 05 0.32625 |           |
```
## policy 2
```

Policy:
|           | 01      > | 02      > | 03      > | 04      < | 05      < |           |
State-value function:
|           | 01 0.51923 | 02 0.69231 | 03   0.75 | 04 0.76923 | 05 0.82692 | 
```
## comparison
```
array([ True, False, False, False, False, False,  True])
```
## conclusion

![WhatsApp Image 2024-03-06 at 11 15 13_c2b4760e](https://github.com/Vivekreddy8360/rl-policy-evaluation/assets/94525701/d2270a02-386a-445b-9e9e-8e4ca1f8a68d)

## RESULT:
Thus a python program was developed successfully
