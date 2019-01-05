import os
import numpy as np
from tqdm import tqdm
from utils.buffer import ReplayBuffer
from utils.utilities import list_to_tensor, partial_obs_2_full_state
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import math
import torch

def get_env():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
       # linux
        env = UnityEnvironment(file_name="./envs/Tennis_Linux/Tennis.x86_64", no_graphics = True)
    elif _platform == "darwin":
       # MAC OS X
       env = UnityEnvironment(file_name="./envs/Tennis.app", no_graphics = False)
    return env

def welcome():
    env = get_env()

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. \n  Each agent observes a state with length: {}'.format(states.shape[0], state_size))

    return env, state_size, action_size, brain_name, num_agents


def maddpg_load_weights(agent, model_dir):
    '''
    :param agent: maddpt agent
    :param dir: top log directory
    :return:
    '''
    for ai in range(2):
        agent.maddpg_agent[ai].actor.load_state_dict(torch.load(model_dir + '/MADDPG/actor_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].critic.load_state_dict(torch.load(model_dir + '/MADDPG/critic_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].target_actor.load_state_dict(torch.load(model_dir + '/MADDPG/target_actor_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].target_critic.load_state_dict(torch.load(model_dir + '/MADDPG/target_critic_{}.pth'.format(ai)))


def maddpg_psne_load_weights(agent, model_dir):
    '''
    :param agent: maddpt agent
    :param dir: top log directory
    :return:
    '''
    for ai in range(2):
        agent.maddpg_agent[ai].actor.load_state_dict(torch.load(model_dir + '/MADDPG_PSNE/actor_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].critic.load_state_dict(torch.load(model_dir + '/MADDPG_PSNE/critic_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].target_actor.load_state_dict(torch.load(model_dir + '/MADDPG_PSNE/target_actor_{}.pth'.format(ai)))
        agent.maddpg_agent[ai].target_critic.load_state_dict(torch.load(model_dir + '/MADDPG_PSNE/target_critic_{}.pth'.format(ai)))


def saddpg_load_weights(agent, model_dir):
    '''
    :param agent:
    :param model_dir:
    :return:
    '''
    agent.saddpg_agent.actor.load_state_dict(torch.load(model_dir + '/SADDPG/actor.pth'))
    agent.saddpg_agent.critic.load_state_dict(torch.load(model_dir + '/SADDPG/critic.pth'))
    agent.saddpg_agent.target_actor.load_state_dict(torch.load(model_dir + '/SADDPG/target_actor.pth'))
    agent.saddpg_agent.target_critic.load_state_dict(torch.load(model_dir + '/SADDPG/target_critic.pth'))





def run_maddpg_agent(env, agent, brain_name, max_t = 1000):
    env_info = env.reset(train_mode=False)
    obs = env_info[brain_name].vector_observations
    t = 0
    while True:

        # explore = only explore for a certain number of episodes
        actions = agent.act(list_to_tensor(obs.tolist()), noise = None)
        actions = torch.stack(actions).detach().numpy()

        # step forward one frame
        env_info = env.step(actions)[brain_name]
        next_obs = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        if (t + 1) == max_t:
            break
        else:
            obs = next_obs



def run_maddpg_psne_agent(env, agent, brain_name, max_t = 1000):
    env_info = env.reset(train_mode=False)
    obs = env_info[brain_name].vector_observations
    t = 0
    while True:

        # explore = only explore for a certain number of episodes
        actions = agent.act(list_to_tensor(obs.tolist()), use_perturbed_actor=False, noise_delta = None)
        actions = torch.stack(actions).detach().numpy()

        # step forward one frame
        env_info = env.step(actions)[brain_name]
        next_obs = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        if (t + 1) == max_t:
            break
        else:
            obs = next_obs

def run_saddpg_agent(env, agent, brain_name, max_t = 1000):
    env_info = env.reset(train_mode=False)
    obs = env_info[brain_name].vector_observations
    t = 0
    while True:
        actions = []
        for ai in range(2):
            _obs = obs.tolist()[ai]
            _obs = torch.tensor(_obs, dtype=torch.float)
            a = agent.act(_obs, noise=None)
            actions.append(a.detach().numpy())

        # step forward one frame
        env_info = env.step(actions)[brain_name]
        next_obs = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        if (t + 1) == max_t:
            break
        else:
            obs = next_obs

    pass



def main(model_type, max_t):
    # max_t = 500
    # model_type = 'MADDPG_PSNE'
    env, state_size, action_size, brain_name, num_agents = welcome()

    if model_type == 'MADDPG':
        from agents.maddpg import MADDPG
        agent = MADDPG(n_agents = num_agents,
                        partial_obs_size = state_size,
                        action_size = action_size,
                        full_obs_size = state_size*num_agents,
                        logger=None)

        maddpg_load_weights(agent, './model_dir')
        run_maddpg_agent(env, agent, brain_name, max_t)

    elif model_type == 'SADDPG':
        from agents.saddpg import SADDPG
        agent = SADDPG(partial_obs_size=state_size,
                        action_size=action_size,
                        full_obs_size=state_size * num_agents,
                        logger=None)
        saddpg_load_weights(agent, './model_dir')
        run_saddpg_agent(env, agent, brain_name, max_t)

    elif model_type == 'MADDPG_PSNE':
        from agents.maddpg_psne import MADDPG_PSNE

        agent = MADDPG_PSNE(n_agents=num_agents,
                          partial_obs_size=state_size,
                          action_size=action_size,
                          full_obs_size=state_size * num_agents,
                          logger=None)

        maddpg_psne_load_weights(agent, './model_dir')
        run_maddpg_psne_agent(env, agent, brain_name, max_t)

    env.close()



import sys, getopt
cwd = os.getcwd()
def file_exists(dir):
    return os.path.isfile(dir)

if __name__== "__main__":
    model_type = 'MADDPG'
    max_t = 500

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts="m:t:")
    except getopt.GetoptError as ex:
        print('{}: run.py -m <model_type: [MADDPG (default)|MADDPG_PSNE|SADDPG]> -t <number of simu steps, 500 (default)>'.format(str(ex)))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-m",):
            model_type = str(arg)
            assert model_type in ['MADDPG', 'MADDPG_PSNE', 'SADDPG'], 'Model type unrecognized: [MADDPG|MADDPG_PSNE|SADDPG]'
        elif opt in ("-t"):
            max_t = int(str(arg))
            assert max_t > 0, 'Max time must be a positive integer'

    main(model_type, max_t)
