
import tensorflow as tf
import  numpy as np
import os
import csv
import matplotlib as plt
from matplotlib.colors import ListedColormap
import pandas as pd


def generate_hyperparameters_sets(
  nbr_sets,
  learning_rate_low = 0.0005,
  learning_rate_high = 0.005,
  gamma_low =0.5,
  gamma_high = 1,
  beta_v_low = 0.1,
  beta_v_high = 0.9,
  beta_e_low = 0.005,
  beta_e_high = 0.5,
  search = 'random'):
  
  if search == 'grid':
    learning_rates = np.arange(learning_rate_low, learning_rate_high, nbr_sets)
    gammas = np.arange(gamma_low, gamma_high, nbr_sets)
    betas_v = np.arange(beta_v_low, beta_v_high, nbr_sets)
    betas_e = np.arange(beta_e_low, beta_e_high, nbr_sets)
  
  if search == 'random':
    learning_rates = np.random.uniform(learning_rate_low, learning_rate_high, nbr_sets)
    gammas = np.random.uniform(gamma_low, gamma_high, nbr_sets)
    betas_v = np.random.uniform(beta_v_low, beta_v_high, nbr_sets)
    betas_e = np.random.uniform(beta_e_low, beta_e_high, nbr_sets)

    
  sets = []
  for i in range(nbr_sets):
    set = []
    set.append(learning_rates[i])
    set.append(gammas[i])
    set.append(betas_v[i])
    set.append(betas_e[i])

    sets.append(set)
  
  return sets

def write_behavior(behavior_path, behavior):
  '''behavior is a list built as follows : 
  [rwd_prob_action0, rwd_prob_action1, action_1, ..., action_t] '''
  if not os.path.exists(behavior_path):
    os.makedirs(behavior_path)  

  with open(behavior_path+'behavior.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(behavior)


def write_training_reward(training_reward_path, episode, reward):
  if not os.path.exists(training_reward_path):
    os.makedirs(training_reward_path)  

  with open(training_reward_path+'training_reward.csv', 'w') as f:
    writer = csv.writer(f)
    row  = [episode, reward]
    writer.writerow(row)

  
def create_config_file(
    experience_file_path, experience_name, seeds, Training, 
    max_episodes_training, num_actions, num_hidden_units, task, learning_rate, beta_e, beta_v):

    if not os.path.exists(experience_file_path):
      os.makedirs(experience_file_path)
    with open(experience_file_path+'config.txt', 'w') as config_file:
        print(f"Experience name: {experience_name}", file = config_file)
        print(f"seeds: {seeds}\n", file = config_file)
        print(f"Training: {Training}", file = config_file)
        print(f"max_episodes_training: {max_episodes_training}\n", file = config_file)
        print(f"learning_rate: {learning_rate}\n", file = config_file)
        print(f"beta_e: {beta_e}", file = config_file)
        print(f"beta_v: {beta_v}\n", file = config_file)
        print(f"num_actions: {num_actions}", file = config_file)
        print(f"num_hidden_units: {num_hidden_units}\n", file = config_file)
        print(f"task: {task}\n", file = config_file)
        
        

    
def get_n_step_return(
    rewards: tf.Tensor,
    values: tf.Tensor,
    n: int,
    gamma: float):
    '''Fonction qui retourne R_t, le gamma utilisé est celui préconisé par 
    Wang et al. (2018), Methods/Simulation1
    Version AVEC bootstrap (utilisation de la valeur prédite au dernier step
    comme point de départ)
    '''
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    gamma = tf.cast(gamma, tf.float32)
    # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
    rewards = rewards[::-1]
    values =  values[::-1]

    # values is inverted
    discounted_sum = values[0]
    for i in tf.range(n):
        discounted_sum = rewards[i] + gamma * discounted_sum        
# I think it is a typo in the article, I put it above in defining discounted_sum
#        discounted = discounted_sum +  values[n-1]* tf.pow(tf.constant(gamma, dtype = tf.float32),tf.cast(n,tf.float32))
        returns = returns.write(i, discounted_sum)
    
    return returns.stack()

#################################################################
#### Deprecated Zone
#################################################################

# not used anymore
def get_input_time_step(reward, action): #reward et action sont deux tenseurs "scalaires"
    action, reward = tf.reshape(action, (1,)), tf.reshape(reward, (1,))
    
    multiple = tf.constant([3])
    action_tensor = tf.tile(action, multiple)
    reward_tensor = tf.tile(reward, multiple)
    action_tensor, reward_tensor = tf.cast(action_tensor, tf.float32), tf.cast(reward_tensor, tf.float32)

    t1 = tf.math.multiply(reward_tensor, tf.constant([1, 0, 0], shape = (3,), dtype = tf.float32))
    t2 = tf.math.multiply(action_tensor, tf.constant([0,-1, 1], shape = (3,), dtype = tf.float32))
    t3 = tf.add(t1, t2)

    input = tf.add(t3, tf.constant([0, 1, 0], shape = (3,), dtype = tf.float32))
    input = tf.reshape(input, (1,3))
    input = tf.expand_dims(input, 0)
    return input

# not used anymore
def get_input(prev_input, r, a):
  input_new_time_step = get_input_time_step(r, a)
  input = tf.concat([prev_input, input_new_time_step], 1)
  return input

# not used anymore
def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float):
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  return returns

# not used anymore
def get_losses(
    action_probs: tf.Tensor,
    rewards: tf.Tensor, 
    values: tf.Tensor, 
    gamma: float = 0.75):
    R_t = get_n_step_return(rewards=rewards,values = values, gamma=gamma)
    delta = R_t - values
    
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    critic_loss = huber_loss(values, R_t)
    
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * delta)

    return actor_loss, critic_loss


def is_hard(p):
    if p < 0.15 or p > 0.85:
        return "easy"
    if p > 0.15 and p < 0.85:
        return "hard"
    else:
        return "no"

def conditional_color(
    lis,
):
    lis = lis.values
    prob = lis[-1]
    if prob < .5:
        for i in range(len(lis[:-1])):
            if lis[i]==0:
                lis[i]=1
            else:
                lis[i]=0
    return lis

def conditioned_color_matrice(
    p: list,
    L: list,
):
    '''
    Create and color a matrice with the input lines, each line being colored conditionally to p.
    
    p: list of lines probabilities
    L: list of lines, a line being a list
    '''
    # creating dataframe L, p
    df = pd.DataFrame(L)
    df["probas"] = p
    df["is_hard"] = df["probas"].apply(lambda x: is_hard(x))
    
    print(df)
    hard_df = df[df["is_hard"]=="hard"].drop(columns=["is_hard"])
    easy_df = df[df["is_hard"]=="easy"].drop(columns=["is_hard"])
    
    # Easy/hard trials
    new_hard_df = pd.DataFrame(list(hard_df.apply(conditional_color, axis=1)))
    new_easy_df = pd.DataFrame(list(easy_df.apply(conditional_color, axis=1)))
    
    # removing p column
    hard_matrice = new_hard_df.T.head(-1).T.values
    easy_matrice = new_easy_df.T.head(-1).T.values
    
    return hard_matrice, easy_matrice


# self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
# self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
# self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)

# self.loss = 0.5 *self.value_loss + self.policy_loss - self.entropy * 0.05