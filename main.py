import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import helper
import tqdm
from matplotlib.colors import ListedColormap
import pandas as pd

rng = np.random.default_rng()

class TwoArmsBanditIndependant:
  
  def __init__(self, p0, p1):
    self.p0 = p0
    self.p1= p1

  def pull(self, a):
    if a == 0:
      reward = rng.random((), dtype=np.float32) < self.p0
    elif a == 1:
      reward = rng.random((), dtype=np.float32) < self.p1
    else:
      raise 
    return reward

class TwoArmsBanditCorrelated:
  
  def __init__(self, p):
    self.p = p

  def pull(self, a):
    if a == 0:
      reward = tf.random.uniform((), dtype=tf.float32) < self.p
    elif a == 1:
      reward = tf.random.uniform((), dtype=tf.float32) < 1 - self.p
    else:
      raise 
    return reward

class Episode:

  def __init__(self, task_label):
    # how many trials in the episode
    #self.nTrials = np.random.randint(50, 101)
    self.nTrials = 100

    if task_label == 'independant':
      probabilities = tf.random.uniform((2,), dtype=tf.float32)
      p0, p1 = probabilities[0], probabilities[1]
      self.task = TwoArmsBanditIndependant(p0, p1)
    
    if task_label == 'correlated':
      p = tf.random.uniform((), dtype=tf.float32)
        
  # an even simpler task: always press right arm
  #      p = tf.constant(0.9, dtype=tf.float32)
      self.task = TwoArmsBanditCorrelated(p)

    
    

class Agent(tf.keras.Model):

    def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
        super().__init__()
        
        self.initialCell = tf.Variable(
                initial_value=tf.zeros((1,num_hidden_units)),
                trainable=True,
                name="initialCell"
                )
        self.initialHidden = tf.Variable(
                initial_value=tf.zeros((1,num_hidden_units)),
                trainable=True,
                name="initialHidden"
                )
        self.LSTM = tf.keras.layers.LSTM(
                num_hidden_units, 
                stateful=True,
                batch_input_shape=(1,1,3),
                name="lstm"
                )
        self.actions = tf.keras.layers.Dense(num_actions, name="actor")
        self.value = tf.keras.layers.Dense(1, name="critic")

    def call(self, inputs: tf.Tensor, initial: bool):
        if initial:
            x = self.LSTM(
                    inputs, 
                    initial_state=[self.initialCell, self.initialHidden]
                    )
        else:
            x = self.LSTM(inputs)
        #return x
        return self.actions(x), self.value(x) #on retourne action_logits_t, value, cell_state (h_t, c_t)


def run_episode(  
    model: tf.keras.Model,
    nTrials: tf.Tensor):
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=nTrials, dynamic_size=False)
  values = tf.TensorArray(dtype=tf.float32, size=nTrials, dynamic_size=False)
  rewards = tf.TensorArray(dtype=tf.float32, size=nTrials, dynamic_size=False)
  
  action_logits_t = tf.zeros((1,2))
  value = tf.zeros((1,))
  episode_entropy = tf.zeros(())
  action0_count=tf.zeros((), dtype=tf.float32)

  p = tf.random.uniform((), dtype=tf.float32)
  
  action = tf.random.uniform((), dtype=tf.int32, maxval=2)
  action1H = tf.reshape(tf.one_hot(action, 2, dtype = tf.float32), (1,2))
  behavior = []
  if action == 0:
    reward = tf.reshape(
            tf.cast(tf.random.uniform((), dtype=tf.float32) < 1-p, tf.float32), 
            (1,1))
  else:
    reward = tf.reshape(
            tf.cast(tf.random.uniform((), dtype=tf.float32) < p, tf.float32), 
            (1,1))

  for t in tf.range(nTrials):
      
    # Store reward
    # for unclear reasons the algorithm needs them with a discrepancy
    # because of the bootstrap thing
    # try here or at the end of the loop
    # not even sure we should do the bootstrap thing in the first place
    rewards = rewards.write(t, tf.squeeze(reward))
      
    inputs = tf.expand_dims(tf.concat([reward, action1H], 1), axis=0)
    
    action_logits_t, value = model(inputs, initial=(t==0))
    
    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)[0, 0]
    behavior.append(tf.get_static_value(action))
    if action == 0:
        action0_count += 1
        
#    tf.print(tf.cast((tf.random.uniform((), dtype=tf.float32) < p), tf.float32).dtype)
    if t < nTrials-1:
      if action == 0:
        reward = tf.reshape(
                tf.cast(tf.random.uniform((), dtype=tf.float32) < 1-p, tf.float32), 
                (1,1))
      else:
        reward = tf.reshape(
                tf.cast(tf.random.uniform((), dtype=tf.float32) < p, tf.float32), 
                (1,1))
            
    action1H = tf.reshape(tf.one_hot(action, 2, dtype = tf.float32), (1,2))

    #Store action probabilities
    action_probs_t = tf.nn.softmax(action_logits_t)

    #behavior = behavior.append(action)
    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])
    
    entropy = -tf.math.reduce_sum(tf.math.multiply(action_probs_t,tf.math.log(action_probs_t + 1e-7)))
    
    episode_entropy += entropy

    # Apply action to the environment to get next state and reward
    # change to enable tf.function
#    if t < nTrials-1:
#      reward = tf.reshape(conditional_rewards[t, action], (1,1))

    #if tf.cast(done, tf.bool):
    # break

  action_probs_tensor = action_probs.stack()
  values_tensor = values.stack()
  rewards_tensor = rewards.stack()
  
#  tf.print()
#  tf.print('proba', episode.task.p)
#  tf.print("reward", tf.reduce_sum(rewards_tensor))
#  tf.print("action 0", action0_count, "action1", 100-action0_count)
#  tf.print("calcul bizarre", episode.task.p * action0_count)
#  tf.print(
#          "expected reward", 
#          episode.task.p * action0_count + (1 - episode.task.p) * (100 - action0_count))



  #return action_probs, episode_entropy,  values, rewards, behavior
  return action_probs_tensor, episode_entropy,  values_tensor, rewards_tensor, action0_count, p, behavior

def compute_loss(
    action_probs: tf.Tensor,
    entropy : tf.Tensor,  
    values: tf.Tensor,
    rewards: tf.Tensor,
    nTrials: tf.Tensor,
    gamma: float,
    beta_v: float,
    beta_e : float) -> tf.Tensor:
  """Computes the combined actor-critic loss."""
  R_t = helper.get_n_step_return(
          rewards=rewards,
          values=values, 
          n=nTrials, 
          gamma=gamma
          )
  delta = R_t - values
  delta_nogradient = tf.stop_gradient(delta)
    
#  huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
#  critic_loss = huber_loss(values, R_t)
      
  critic_loss = 0.5 * tf.reduce_sum(tf.square(delta))
   
  action_log_probs = tf.math.log(action_probs + 1e-7)
      
  # careful with the sign of this one
  #no gradient through temporal difference here
  actor_loss = tf.reduce_sum(action_log_probs * delta_nogradient)
      
  # l'entropie est bien avec un +
  total_loss = -actor_loss + beta_v * critic_loss - beta_e * entropy

  return total_loss, actor_loss, beta_v*critic_loss, beta_e*entropy

@tf.function
def train_step(
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    nTrials: tf.Tensor,
    gamma: float) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
      
    # Run the model for one episode to collect training data
    action_probs, episode_entropy, values, rewards, action0_count, p = run_episode(model, nTrials) 

    # Calculate expected returns
  #returns = tf.expand_dims(helper.get_n_step_return(rewards, values, gamma),1)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values = [
        tf.expand_dims(x, 1) for x in [action_probs, values]] 

    # Calculating loss values to update our network
    loss, actor_loss, critic_loss, entropy_reg = compute_loss(
            action_probs, 
            episode_entropy, 
            values, 
            rewards, 
            nTrials, 
            gamma,
            beta_v,
            beta_e)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  episode_reward = tf.math.reduce_sum(rewards)
  
  return episode_reward, loss, actor_loss, critic_loss, entropy_reg, action0_count, p



# ----------- BEGIN CONFIG ------------- #

# Model to train or to test
num_actions = 2
num_hidden_units = 48
model = Agent(num_actions, num_hidden_units)
  
learning_rate = 0.001
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=learning_rate,
    rho=0.9, 
    momentum=0.0, 
    epsilon=1e-07, 
    centered=False,
    name='RMSprop'
    )

gamma = 0.8 

Training = False
max_episodes_training = 10001
beta_v =  0.006
beta_e = 0.08
                
Testing = True
max_episodes_testing = 500
running_reward = 0


seeds = [1, 2, 3]

load_model = True
load_model_path = './correlated_alpha=0.0005_gamma=0.8beta-v=0.005_beta-e=0_seed3/ckpt/ckpt_episode15000'

#Task
task_label ='correlated'

if load_model:
  model = Agent(num_actions, num_hidden_units)
  print("Loading Model ...")

hyperparameters_tuning = False


# ----------- END CONFIG ------------- #

if Training :
  if hyperparameters_tuning:
    hyperparameters_sets = helper.generate_hyperparameters_sets(50)

  else :
    hyperparameters_sets = [[learning_rate, gamma, beta_v, beta_e]]

  for hyperparameters_set in hyperparameters_sets:
    
    for seed in seeds:

      learning_rate = hyperparameters_set[0]
      gamma = hyperparameters_set[1]
      beta_v = hyperparameters_set[2]
      beta_e = hyperparameters_set[3]

      experience_name = task_label + '_alpha=' + str(learning_rate)[0:6]+ '_gamma='+str(gamma)[0:5]+'beta-v='+str(beta_v)[0:5]+'_beta-e='+str(beta_e)[0:5]+'_seed'+str(seed)
      experience_file_path = './'+ experience_name + '/'
      
      if not os.path.exists(experience_file_path):
        os.makedirs(experience_file_path)

      summary_file_path = experience_file_path+'summary/'
      print("------ summary_" + experience_name+"-------")

      if not os.path.exists(summary_file_path):
            os.makedirs(summary_file_path)

      writer = tf.summary.create_file_writer(summary_file_path)    

      helper.create_config_file(
          experience_file_path, experience_name, seeds, Training, max_episodes_training, 
          num_actions, num_hidden_units, task_label, learning_rate, beta_e, beta_v)



      print("----- START TRAINING -----\n")
      save_model_path = os.path.join(experience_file_path, 'ckpt/')
      if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    # Keep last episodes reward
  #  episodes_reward: collections.deque = collections.deque(maxlen=100)

      ps = []
      action0s = []
      rewards = []
    
      with tqdm.trange(max_episodes_training) as t:
        for i in t:
          episode = Episode(task_label)
          # generate all random results before calling train_step
          # This is working fine and according to the distribution
          nTrials = tf.convert_to_tensor(episode.nTrials, dtype=tf.int32)
        
          episode_reward, loss, actor_loss, critic_loss, entropy_reg, action0_count, p = train_step(
                  model, 
                  optimizer, 
                  nTrials, gamma)
        
          ps.append(p)
          action0s.append(action0_count)
          rewards.append(episode_reward)
          
          with writer.as_default():
              tf.summary.scalar("episode_reward", episode_reward, step=i)
              tf.summary.scalar("total_loss", loss, step=i)
              tf.summary.scalar("actor_loss", actor_loss, step=i)
              tf.summary.scalar("critic_loss", critic_loss, step=i)
              tf.summary.scalar("entropy_regularization", entropy_reg, step=i)
              tf.summary.scalar("nbr_action0", action0_count, step=i)
            
  #      episodes_reward.append(episode_reward)
  #      running_reward = statistics.mean(episodes_reward)    #reward over the last 100 episodes
          
          t.set_description(f'Episode {i}')
          t.set_postfix(
              episode_reward=episode_reward.numpy(), 
  #            running_reward=running_reward,
              total_loss=loss.numpy(),
              actor_loss=actor_loss.numpy(),
              critic_loss=critic_loss.numpy(),
              entropy_reg=entropy_reg.numpy(),
              )
        
        #Show average episode reward each 100 episodes
  #      if i % 100 == 0:
  #        print(f'Episode {i}: average reward: {running_reward}')

        #Save model every 5000 episodes
          if (not i ==0) and i % 5000 == 0:
            model.save_weights(save_model_path+ 'ckpt_episode'+str(i))
            print(f'Model saved at episode {i}')
          


if Testing:
  print("----- LOAD MODEL -----")
  n = 1
  n  = tf.convert_to_tensor(n, dtype=tf.int32)
  a = run_episode(model,n)
  model.load_weights(load_model_path)
  ps = []
  behaviors =[]
  for t in range(50):
    _, _,  _, _, _, p, behavior =run_episode(model, tf.convert_to_tensor(100, dtype=tf.int32))
    ps.append(tf.get_static_value(p))
    behaviors.append(behavior)

  cmap = ListedColormap(['g', 'r'])

  hard_trials, easy_trials = helper.conditioned_color_matrice(ps,behaviors)
  nbr_trials = 15

  plt.imshow(easy_trials[:nbr_trials], cmap=cmap)
  plt.axis('off')
  plt.title('Essais faciles')
  plt.grid(color='k')
  plt.show()

  plt.imshow(hard_trials[:nbr_trials], cmap=cmap)
  plt.title('Essais difficiles')
  plt.axis('off')
  plt.grid()
  plt.show()


