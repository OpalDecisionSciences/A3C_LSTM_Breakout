
# TRaininf the A3C AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from A3C import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    # desynchronize each training agent, using rank to shift each seed with the rank
    # 1 int per agent, when we shift the seed by 1 thread
    # all pseudo random numbers created from the thread will be independent of the other thread
    # seeds are fixed numbers so when we reproduce the experience, we will receive exactly same events
    # deterministic with respect to the seed
    # hence why we desynchronize each training agent by using rank to shift the seed with the rank
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    # each agent of the A3C model has it's own vision of the environment
    # align each agent to one specific version of the environment
    # because each seed determines a different environment, we associate a different seed to each agent
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # prepare input states (images): (1, 42) = black and white, 42 X 42 pixels
    # get numpy array and convert to torch tensor
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0 
    while True:
        episode_length += 1
        # synchronize with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            # reinitialize hidden states and cell states of the lstm of the model
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            # keep old cell states and hidden states
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        # intialize several variables at heart of computations in the training
        values = []
        log_probs = []
        rewards = []
        entropies = []
        # update values of these four variables in steps of exploration
        # for all steps in exploration, get predictions of the model (V(s), Q(s, a), (hx, cx))
        for step in range(params.num_steps):
            # apply model to inputs in batch form (add fake dimension)
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            # generate distribution of probabilities of the input, for action_values
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            # take a random draw of the distribution of probabilities
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            # Play the action to get the transition: new state, new reward, if game is done or not
            state, reward, done = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            # convert the new state into a torch tensor
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        # initialize commulative reward as torch tensor
        R = torch.zeros(1, 1)
        # commulative reward = value of last state reached by the shared network
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        # initialize losses, related to predicitions of agent and predictions of critic
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        # Generalized advantage estimation (GAE) - the advantage of playing action a by observing the state s
        # difference between q-values and value of v function applied to the state s
        # A(a,s) = Q(a, s) - V(s)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            # R = commulative reward
            # R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1} + gamma^nb_steps * V(last_state)
            R = params.gamma *R + rewards[i]
            advantage = R - values[i]
            # loss generated by the predictions of the value of the v function output by the critic
            # when we play the optimal action, we get the stationary state with Q optimal of the optimal action a star played in the state s equals optimal value of v star
            # Q*(a*,s) = V*(s)
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # We need the gae to compute the policy_loss, which requires gae
            # to get the gae, we need the value of the temporal difference of the state values 
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            # gae = sum_i (gamma*tau)^i * TD(i)
            gae = gae * params.gamma * params.tau + TD
            # policy_loss = - sum_i log(softmax probabilities of the action at step i: pi_i)*gae + 0.01*H_i
            # maximize the probability of playing the action that will maximize the advantage
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
        optimizer.zero_grad()
        # backward propagation
        (policy_loss + 0.5 * value_loss).backward()
        # prevent gradient from taking extremely large values and degenerate the algorithm
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        # perform optimization ste to reduce the losses
        optimizer.step()




