import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        # TODO init rbf encoder
        # campionate 10,000 random states
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        # StandardScaler normalize data and compute mean and stddev
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        # apply scaling to the examples

        # create 4 RBFSample with different gammas [5.0 , 2.0 , 1.0 , 0.5] to capture different details
        # we use FeatureUnion because it works well in parallel
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rfb1", RBFSampler(gamma=5.0, n_components=50)),
            ("rfb2", RBFSampler(gamma=2.0, n_components=50)),
            ("rfb3", RBFSampler(gamma=1.0, n_components=50)),
            ("rfb4", RBFSampler(gamma=0.5, n_components=50))
        ])
        # we have a total of 200 hundred features
        
        # fit the featurizer with scaled observations
        scaled_examples = self.scaler.transform(observation_examples)
        self.featurizer.fit(scaled_examples)
        

    def encode(self, state): # modify
        # TODO use the rbf encoder to return the features
        scaled = self.scaler.transform([state])
        features = self.featurizer.transform(scaled)
        return features[0]
    @property
    def size(self): # modify
        # TODO return the number of features
        dummy_state = self.env.observation_space.sample()
        features = self.encode(dummy_state)
        return len(features)

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape) * 0.001
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    # Q returns a vector of size (n_actions,1)    
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        # TODO update the weights
        
        # get Q-values and greedy action for Watkins's check
        Q_s = self.Q(s_feats)
        greedy_action = Q_s.argmax()
        
        # compute TD target (off-policy)
        Q_current = Q_s[action, 0]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q(s_prime_feats).max()
        
        # compute TD error and clip it for stability
        delta_t = td_target - Q_current
        delta_t = np.clip(delta_t, -1.0, 1.0)
        
        # update traces: reset if non-greedy (Watkins), otherwise decay
        if action != greedy_action:
            self.traces.fill(0)
        else:
            self.traces *= self.gamma * self.lambda_
            
        # replacing traces
        self.traces[action] = s_feats
        self.traces = np.clip(self.traces, -1.0, 1.0)
        
        # update weights
        self.weights += self.alpha * delta_t * self.traces
        
        
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
