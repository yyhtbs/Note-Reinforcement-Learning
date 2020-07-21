# Policy Gradient

PPO: on-policy -> off-policy

## Baic Components
Each Policy Gradient contains Actor, Environment and Reward Function. 
![](./Hungyi-Lee-L1-Image/Actor-Env-Reward.png)

Actor is controllable however, environment and reward function are given cannot be changed. 

## Policy of Actor

Policy $\pi$ is a neural network (NN) in DRL, with paramter $\theta$.
The policy takes the observations (e.g. state) to output the "score" of executing each action. In the implementation, the score is the probability of executing each action. 

The actor takes the action based on the probability (i.e. via inverse sampling).

## Relationship between Actor, Env and Reward

Env $\rightarrow$ $s_1$ $\rightarrow$ Actor $\rightarrow$ $a_1$ $\rightarrow$ Env $\rightarrow$ $s_2$ $\rightarrow$ $\cdots$

The interations between the actor and the environment formulate a trajectory $\tau = \{s_1, a_1, s_2, a_2, \cdots, s_T, a_T\}$ which contains a sequence of state-action pairs. 

Based on the current action with a policy configuration $\theta$, the probability of observing the trajectory $\tau$ is, 

$$p_{\theta}(\tau) = p(s_1)p_\theta(s_1|s_1)p(s_2|s_1, a_1)p_\theta(a_2|s_2) \cdots\\
=p(s_1)\prod^{T}_{t = 1}p_\theta(s_t|s_t)p(s_{t+1}|s_t, a_t)$$

After executing an action $a$ at a given state $s$, the actor will receive a reward $(s_1, a_1) \rightarrow r_1$.

For each trajectory $\tau$, the total reward is $R(\tau) = \sum_{t \in \tau}r_t$. Because the trajectory is a random variable, the reward of given an actor with $\theta$, the total reward $\bar{R}_\theta$ is the expectation of the trajectory rewards $R(\tau)$.

$$\bar{R}_\theta = \sum_{\tau}{R(\tau)p_\theta(\tau)} = \mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)]$$
Here $\tau\sim p_\theta(\tau)$ means sampling actor trajectory from the environment.

## Training

The object function is to maximise the expected reward. 

Method: Gradient Ascent, to calculate the gradient $\nabla\bar{R}_\theta$ of expected reward.

$$\nabla\bar{R}_\theta = \sum_{\tau}{R(\tau)\nabla p_\theta(\tau)}$$

Here $R(\tau)$ is the observed reward, does not need to be differentiable (it can be even be a blackbox system). 

$$\nabla\bar{R}_\theta = \sum_{\tau}{R(\tau)p_\theta(\tau) \frac{\nabla p_\theta(\tau)}{p_\theta(\tau)}} = \sum_{\tau}{R(\tau)p_\theta(\tau)}\nabla \log p_\theta(\tau)\\ =\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)\nabla \log p_\theta(\tau)]$$

NOTE: A common formula $\nabla f(x) = f(x) \nabla\log f(x)$

However, it is difficult to calculate the expectation (it needs enumerating all possible trajectories), so a practical solution is to sample $\tau$ and approximate its distribution. 

$$\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)\nabla \log p_\theta(\tau)] 
\approx \frac{1}{N}\sum^{N}_{n = 1}R(\tau^n)\nabla \log p_\theta(\tau^n)$$

Here $n = [1 \dots N]$ corresponds to the trajectory samples. 

Let us further expand the formula, 

$$\frac{1}{N}\sum^{N}_{n = 1}R(\tau^n)\nabla \log p_\theta(\tau^n)
= \frac{1}{N}\sum^{N}_{n = 1}R(\tau^n)\nabla \log \prod_{t =1}^{T_N} p_\theta(a_t^n|s_t^n)
= \frac{1}{N}\sum^{N}_{n = 1}\sum_{t=1}^{T_N}R(\tau^n)\nabla \log p_\theta(a_t^n|s_t^n)$$

## Policy Gradient

The parameters of NN is updated to maximise $\nabla\bar{R}_\theta$ whereas the output of NN is the probability of executing an action $a$ at a given state $s$. 

The update rule is $\theta \leftarrow \theta + \eta \nabla\bar{R}_\theta$.

In practice, the actor needs to collect a number of trajectories from interactions in which each trajectory is a sequence of state-action pairs and rewards.  

## Implementation using DNN

Training the DRL using policy gradient is very similar to a classification problem in which the object is to maximise the reward function for inputs $s$ and outputs $a$, 
$$\frac{1}{N}\sum^{N}_{n = 1}\sum^{T_N}_{t = 1}R(\tau^n)\log p_\theta(a^b_t|s^b_t)$$

This object function will automatically derive the gradient $\nabla\bar{R}_\theta$

## Tip 1: Add Baseline

It is not good if the reward is always positive. So we need a baseline to justify whether a action is sufficiently good to be a positive reward. 

$$\theta \leftarrow \theta + \eta \nabla\bar{R}_\theta\\ \nabla\bar{R}_\theta = \frac{1}{N}\sum^{N}_{n = 1}\sum^{T_N}_{t = 1}(R(\tau^n) - b)\nabla\log p_\theta(a^b_t|s^b_t) ~~~~ b\approx\mathbf{E}(R(\tau))$$

## Tip 2: Assign Suitable Credit









