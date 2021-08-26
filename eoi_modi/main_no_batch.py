import numpy as np
import sys, os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow.compat.v1.keras.utils import to_categorical
from model import *
from buffer import ReplayBuffer
from config import *
from utils import *
import multiagent.scenarios as scenarios
from env_wrapper import env_wrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

alpha = float(sys.argv[1])

# load scenario from script
scenario = scenarios.load("simple_spread.py").Scenario()
# create world
world = scenario.make_world()
env = env_wrapper(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
env.discrete_action_input = True
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
feature_space = env_info["obs_shape"]
state_space = env_info["state_shape"]
observation_space = feature_space + n_ant

buff = ReplayBuffer(capacity, state_space, observation_space, n_actions, n_ant)
agents = Agent(sess, observation_space, n_actions, state_space, n_ant, alpha)
eoi_net = intrisic_eoi(feature_space, n_ant)
get_eoi_reward = build_batch_eoi(feature_space, eoi_net, n_ant)

agents.update()

q = np.ones((n_ant, batch_size, n_actions))
next_a = np.ones((n_ant, batch_size, n_actions))
feature = np.zeros((batch_size, feature_space))
feature_positive = np.zeros((batch_size, feature_space))

f = open(sys.argv[1] + '_' + sys.argv[2] + '.txt', 'w')
batch_size = 1

def test_agent():
    episode_reward = 0

    for e_i in range(20):
        env.reset()
        obs = get_obs(env.get_obs(), n_ant)
        mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
        terminated = False
        cnt = 0
        while terminated == False:
            acts = []
            cnt += 1
            if (cnt > 50):
                terminated = True
            outs = agents.acting([np.array([obs[i]]) for i in range(n_ant)])
            for i in range(n_ant):
                a = np.argmax(outs[i][0] - 9e15 * (1 - mask[i]))
                acts.append(a)

            new_obs_n, reward_n, done_n, info_n = env.step(acts)
            reward = reward_n[0]
            episode_reward += reward
            obs = get_obs(env.get_obs(), n_ant)
            mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
    return episode_reward / 20



while i_episode < n_episode:
    i_episode += 1
    print("i_episode=")
    print(i_episode)
    if i_episode > 40:
        epsilon -= 0.005
        if epsilon < 0.05:
            epsilon = 0.05
    env.reset()
    obs = get_obs(env.get_obs(), n_ant)
    state = env.get_state()
    mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
    terminated = False
    episode_reward = 0
    win = 0
    cnt = 0
    while terminated == False:
        action = []
        acts = []
        cnt += 1
        if (cnt > 1):
            terminated = True
        outs = agents.acting([np.array([obs[i]]) for i in range(n_ant)])
        action = []
        for i in range(n_ant):
            if np.random.rand() < epsilon:
                avail_actions_ind = np.nonzero(mask[i])[0]
                a = np.random.choice(avail_actions_ind)
            else:
                a = np.argmax(outs[i][0] - 9e15 * (1 - mask[i]))
            acts.append(a)
            action.append(to_categorical(a, n_actions))
        new_obs_n, reward_n, done_n, info_n = env.step(acts)
        reward = reward_n[0]
        episode_reward += np.array(reward)
        next_obs = get_obs(new_obs_n, n_ant)
        next_state = env.get_state()
        next_mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
        buff.add(np.array(obs), action, reward, np.array(next_obs), state, next_state, mask, next_mask, terminated)
        obs = next_obs
        state = next_state
        mask = next_mask
    print(episode_reward)
    sum_reward += episode_reward

    if i_episode % 200 == 0:
        log_r = test_agent()
        h = str(int(i_episode / 200)) + ': ' + sys.argv[1] + ': ' + sys.argv[2] + ': ' + str(
            sum_reward / 200) + ': ' + str(sum_win / 200) + ': ' + str(log_r)
        print(h)
        f.write(h + '\n')
        f.flush()
        sum_reward = 0
        sum_win = 0

    if i_episode < 100:
        continue

    samples, positive_samples = buff.getObs(1)
    feature_label = np.random.randint(0, n_ant, 1)
    for i in range(1):
        feature[i] = samples[feature_label[i]][i][0:feature_space]
        feature_positive[i] = positive_samples[feature_label[i]][i][0:feature_space]
    sample_labels = to_categorical(feature_label, n_ant)
    positive_labels = eoi_net.predict(feature_positive, batch_size=1)
    eoi_net.fit(feature, sample_labels + beta_1 * positive_labels, batch_size=1, epochs=1, verbose=0)

    for e in range(epoch):

        o, a, r, next_o, s, next_s, mask, next_mask, d = buff.getBatch(1)

        q_q = agents.batch_q([o[i] for i in range(n_ant)])
        next_q_q = agents.batch_q_tar([next_o[i] for i in range(n_ant)])
        eoi_r = get_eoi_reward.predict([o[i][:, 0:feature_space] for i in range(n_ant)], batch_size=1)
        for i in range(n_ant):
            best_a = np.argmax(next_q_q[i] - 9e15 * (1 - next_mask[i]), axis=1)
            next_a[i] = to_categorical(best_a, n_actions)
            q[i] = q_q[i + n_ant]
            for j in range(1):
                q[i][j][np.argmax(a[i][j])] = gamma * (1 - d[j]) * next_q_q[i + n_ant][j][best_a[j]] + eoi_r[i][j]
        agents.train_critics(o, q)

        q_target = agents.Q_tot_tar.predict(
            [next_o[i] for i in range(n_ant)] + [next_a[i] for i in range(n_ant)] + [next_s], batch_size=1)
        q_target = r + q_target * gamma * (1 - d)
        agents.train_qmix(o, a, s, mask, q_target)

    if i_episode % 5 == 0:
        agents.update()
