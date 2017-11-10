from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=26, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

class DQN(object):

    def __init__(self, env):
        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.98
        self.eps_start = 1.0
        self.eps_decay = 0.999 # in episodes
        # If using a target network
        self.clone_steps = 10000
        self.learning_rate = .001

        # Create replay memory buffer
        self.replay_memory = ReplayMemory(100000)
        # Don't start analyzing replay memory until we have a certain number
        self.min_replay_size = 1000

        # Create the training and target networks
        self.build_model()

        # Initialize all of the variables in the networks
        self.sess.run(tf.global_variables_initializer())

        # Set values of train network to the target network
        self.update_operations = []
        self.create_copy_operations()
        # define your update operations here...

        self.num_episodes = 0
        self.num_steps_with_no_update = 0

        self.saver = tf.train.Saver(tf.trainable_variables())

    # Copy train network values to target network
    def create_copy_operations(self):
        trainable_variables = tf.trainable_variables('train')
        target_variables = tf.trainable_variables('target')

        # For each variable, create an assign operation
        for i in range(0, len(trainable_variables)):
            self.update_operations.append(target_variables[i].assign(trainable_variables[i]))

        # Perform the assign operations, copying everything from the train variables
        #   to the target variables
        self.sess.run(self.update_operations)

        return self.update_operations

    def build_model(self):

        # Define input tensor which takes 8 input values
        self.observation_input = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])

        # Action tensor
        self.action_input = tf.placeholder(tf.float32, [None, env.action_space.n])

        # Q Value tensor
        self.target_q_val = tf.placeholder(tf.float32, [None,])


        with tf.variable_scope('train'):
            train_net = tf.contrib.layers.fully_connected(self.observation_input, 16, activation_fn=tf.nn.relu)
            train_net = tf.contrib.layers.fully_connected(train_net, 32, activation_fn=tf.nn.relu)
            self.train_network = tf.contrib.layers.fully_connected(train_net, env.action_space.n, activation_fn=None)

        with tf.variable_scope('target'):
            target_net = tf.contrib.layers.fully_connected(self.observation_input, 16, activation_fn=tf.nn.relu)
            target_net = tf.contrib.layers.fully_connected(target_net, 32, activation_fn=tf.nn.relu)
            self.target_network = tf.contrib.layers.fully_connected(target_net, env.action_space.n, activation_fn=None)

        # NOTE: Here we used some different loss functions including softmax and some
        #       other custom loss functions that we thought were better estimators
        action_q_val = tf.reduce_sum(tf.multiply(self.train_network, self.action_input), reduction_indices=1)
        q_val_error = tf.reduce_mean(tf.losses.huber_loss(self.target_q_val, action_q_val))


        # NOTE: Here tried many different optimizers including gradient decend and RMS with
        #       various values for learning rate
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(q_val_error)

    def select_action(self, obs, evaluation_mode=False):

        # Transform the observation to be of the correct input for the network
        obs = np.reshape(np.asarray(obs, dtype=np.float32), (1, 8))

        # If evaluating, return our best choice
        if evaluation_mode:
            act = np.argmax(self.sess.run(self.target_network, feed_dict={self.observation_input: obs}))
            return act

        # With some probability, choose a random action, else return the best answer
        randomValue = np.random.random()
        if randomValue < self.eps_start:
            randomAction = env.action_space.sample()
            return randomAction
        else:
            return np.argmax(self.sess.run(self.train_network, feed_dict={self.observation_input: obs}))


    def update(self):

        if len(self.replay_memory.memory) > self.min_replay_size:

            transitions = self.replay_memory.sample(self.batch_size)
            states, actions, next_states, rewards, is_terminal_list = Transition(*zip(*transitions))
            actions = self.encode_action(actions) # Encode each of the actions as a 1 hot vector

            # Reshape the arrays to be input into the model
            states = np.asarray(states, dtype=np.float32)
            states = np.reshape(states, (self.batch_size, 8))
            next_states = np.asarray(next_states, dtype=np.float32)
            next_states = np.reshape(next_states, (self.batch_size, 8))
            is_terminal_list = np.array(is_terminal_list)
            rewards = np.array(rewards)

            target = rewards + (1 - is_terminal_list) * self.gamma * np.max(
                self.sess.run(self.target_network, feed_dict={self.observation_input: next_states}))

            self.sess.run(self.update_op, feed_dict={
                self.observation_input: states,
                self.action_input: actions,
                self.target_q_val: target})


    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()

		# Here we tried storing the observations for a whole episode and then given the last reward value of the
		#	episode we would back propogate that value using gamma to diminish the value over the states.  This
		#	was our attempt to encode in previous reward values a better temporal nature.
        episodeObservations = []
        rewardObservations = []
        while not done:

            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)

            #episodeObservations += [(obs, action, next_obs, done)]
            #rewardObservations += [reward]

            self.replay_memory.push(obs, action, next_obs, reward, done)
            self.update()

            obs = next_obs
            self.num_steps_with_no_update += 1

            if done:
                break

        #self.store_observations(episodeObservations, rewardObservations)

        # If the number of steps is above a threshold then update the target model
        if self.num_steps_with_no_update > self.clone_steps:
            self.sess.run(self.update_operations)
            self.num_steps_with_no_update = 0
            #print("Updated target model")

        self.eps_start *= self.eps_decay
        self.num_episodes += 1

    def store_observations(self, episodeObservations, rewardObservations):

        #Update the reward values taking into account the end reward, iterate backwards through the
		#	frames of the episode
        for t in reversed(range(0, len(rewardObservations) - 1)):
            rewardObservations[t] = self.gamma * rewardObservations[t + 1] + rewardObservations[t]

		# Normalize
        rewardObservations -= np.mean(rewardObservations)
        rewardObservations /= np.std(rewardObservations)

        # Now that the rewards are updated, write to replay mem using the new reward value
        for i in range(0, len(episodeObservations)):
            obs, action, next_obs, done = episodeObservations[i]
            reward = rewardObservations[i]
            self.replay_memory.push(obs, action, next_obs, reward, done)


    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        done = False
        obs = env.reset()
        while not done:
            if self.num_episodes > 1000:
                env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode (", self.num_episodes, "): ", total_reward, "Epsilon: ", self.eps_start)
        if save_snapshot:
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)


    def encode_action(self, actions):
        '''Returns 1 hot for action val'''
        value_array = np.zeros(shape=(self.batch_size, 4))
        for index, action in enumerate(actions):

            # Append one-hot numpy vector to value_array
            value_array[index] = np.array([int(i == action) for i in range(4)])

        return value_array

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    for i in range(100):
        dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)