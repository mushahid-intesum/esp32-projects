import tensorflow as tf
import numpy as np
from tensorflow import keras
from expert import *
from arbiter import *
from constants import *
import wandb

wandb.login()

wandb.init(
    project="MCU MoE-ish",
    name="MCU MoE-ish",
    config={
        "algorithm": "MCU MoE-ish",
        "learning_rate": 3e-4,
        "episodes": 1000,
        "epsilon_clip": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "ppo_epochs": 4
    }
)

EXPERT_LABELS = {
    0: list(range(0, 10)),
    1: list(range(10, 20)),
    2: list(range(20, 30)),
    3: list(range(30, 40)),
    4: list(range(40, 50))
}

def get_expert_for_label( label):
    for expert_id, labels in EXPERT_LABELS.items():
        if label in labels:
            return expert_id
    return None
    
def label_to_expert_label(label, expert_id):
    expert_labels = EXPERT_LABELS[expert_id]
    return expert_labels.index(label)

def compute_discounted_returns(rewards, gamma=0.99):
    returns = []
    running_return = 0.0
    
    for r in reversed(rewards.numpy()):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    
    return tf.constant(returns, dtype=tf.float32)

def train_arbiter_rl(arbiter,
                    experts,
                    train_data,
                    episodes=1000,
                    algorithm='PPO'):
    print(f"\nTraining Arbiter with {algorithm}...")
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0003)
   
    # Reward structure
    CORRECT_REWARD = 1.0
    INCORRECT_PENALTY = -0.5
    NO_DETECTION_REWARD = -0.1
    EFFICIENCY_BONUS = 0.2
    
    # PPO-specific hyperparameters
    epsilon_clip = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    ppo_epochs = 4
       
    
    for episode in range(episodes):
        episode_rewards = []
        episode_actions = []
        episode_logits = []
        episode_values = []
        episode_latents = []
        episode_old_probs = []
        episode_inputs = []
        
        # === EXPERIENCE COLLECTION ===
        for x_batch, y_batch in train_data:
            batch_size = tf.shape(x_batch)[0]
            latent, policy_logits, values = arbiter(x_batch, training=False)
            
            actions = sample_action(policy_logits)
            actions = tf.squeeze(actions, axis=1)
            
            # Store old action probabilities for PPO
            action_probs = tf.nn.softmax(policy_logits)
            action_masks = tf.one_hot(actions, NUM_ACTIONS)
            old_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            episode_old_probs.append(old_probs)
            
            # Calculate rewards
            rewards = []
            
            for i in range(batch_size):
                action = actions[i].numpy()
                true_label = y_batch[i].numpy()
                
                if action == 0:
                    reward = NO_DETECTION_REWARD
                else:
                    expert_id = action - 1
                    correct_expert = get_expert_for_label(true_label)
                    
                    if expert_id == correct_expert:
                        expert_latent = tf.expand_dims(latent[i], 0)

                        pred = experts[expert_id](expert_latent, training=False)
                        expert_label = label_to_expert_label(true_label, expert_id)
                        
                        if tf.argmax(pred[0]).numpy() == expert_label:
                            reward = CORRECT_REWARD + EFFICIENCY_BONUS
                        else:
                            reward = INCORRECT_PENALTY
                    else:
                        reward = INCORRECT_PENALTY
                
                rewards.append(reward)
                
         
            episode_rewards.append(rewards)
            episode_actions.append(actions)
            episode_logits.append(policy_logits)
            episode_values.append(values)
            episode_latents.append(latent)
            episode_inputs.append(x_batch)
        
        if algorithm == 'PPO':
            for ppo_epoch in range(ppo_epochs):
                with tf.GradientTape() as tape:
                    total_policy_loss = 0
                    total_value_loss = 0
                    total_entropy = 0
                    
                    for batch_idx in range(len(episode_rewards)):
                        rewards = tf.constant(episode_rewards[batch_idx], dtype=tf.float32)
                        actions = episode_actions[batch_idx]
                        latent = episode_latents[batch_idx]
                        old_probs = episode_old_probs[batch_idx]
                        input = episode_inputs[batch_idx]

                        new_latents, new_policy, new_values = arbiter(input)
                        new_probs_all = tf.nn.softmax(new_policy)
                        
                        returns = compute_discounted_returns(rewards, gamma=0.99)
                        
                        advantages = returns - tf.squeeze(new_values)
                        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
                        
                        action_masks = tf.one_hot(actions, NUM_ACTIONS)
                        new_probs = tf.reduce_sum(new_probs_all * action_masks, axis=1)
                        
                        ratio = new_probs / (old_probs + 1e-10)
                        surr1 = ratio * advantages
                        surr2 = tf.clip_by_value(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
                        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                        
                        value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(new_values)))
                        
                        entropy = -tf.reduce_mean(
                            tf.reduce_sum(new_probs_all * tf.math.log(new_probs_all + 1e-10), axis=1)
                        )
                        
                        total_policy_loss += policy_loss
                        total_value_loss += value_loss
                        total_entropy += entropy
                    
                    total_loss = total_policy_loss + value_coef * total_value_loss - entropy_coef * total_entropy
                
                grads = tape.gradient(total_loss, arbiter.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)  # Gradient clipping
                optimizer.apply_gradients(zip(grads, arbiter.trainable_variables))
            
                avg_reward = np.mean([r for batch in episode_rewards for r in batch])
                print(f"Episode {episode}: Avg Reward={avg_reward:.4f}, "
                        f"Loss={total_loss.numpy():.4f}")
                
                wandb.log({
                    "episode": episode,
                    "episode_rewards": avg_reward,
                    "policy_loss": total_policy_loss.numpy(),
                    "value_loss": total_value_loss.numpy(),
                    "entropy": total_entropy.numpy(),
                    "total_loss": total_loss.numpy()
                })
        
       
        else:  # REINFORCE (default)
            # REINFORCE: Basic policy gradient with baseline
            with tf.GradientTape() as tape:
                total_policy_loss = 0
                total_value_loss = 0
                
                for batch_idx in range(len(episode_rewards)):
                    rewards = tf.constant(episode_rewards[batch_idx], dtype=tf.float32)
                    actions = episode_actions[batch_idx]
                    logits = episode_logits[batch_idx]
                    values = episode_values[batch_idx]
                    
                    returns = rewards
                    advantages = returns - tf.squeeze(values)
                    
                    # Policy loss
                    action_probs = tf.nn.softmax(logits)
                    action_masks = tf.one_hot(actions, NUM_ACTIONS)
                    selected_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
                    log_probs = tf.math.log(selected_probs + 1e-10)
                    policy_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
                    
                    # Value loss
                    value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
                    
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                
                total_loss = total_policy_loss + 0.5 * total_value_loss
            
            grads = tape.gradient(total_loss, arbiter.trainable_variables)
            optimizer.apply_gradients(zip(grads, arbiter.trainable_variables))
            
            if episode % 50 == 0:
                avg_reward = np.mean([r for batch in episode_rewards for r in batch])
                print(f"Episode {episode}: Avg Reward={avg_reward:.4f}, "
                      f"Loss={total_loss.numpy():.4f}")
                
def train_expert_supervised(arbiter, expert, train_data, val_data, epochs=50):
    """Train expert models with supervised learning"""    
    print(f"\nTraining Expert")
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # Get only the trainable variables for the latent encoder
    # Exclude policy and value head variables
    arbiter_trainable_vars = [v for v in arbiter.trainable_variables 
                              if 'policy' not in v.name and 'value' not in v.name 
                              and 'dense' not in v.name]
    
    for epoch in range(epochs):
        train_loss = []
        train_acc = [] 
        val_loss = []
        val_acc = [] 
        
        # Training loop
        for x_batch, y_batch in train_data:
            with tf.GradientTape() as tape:
                # Get latent representation from arbiter
                latent, _, _ = arbiter(x_batch, training=False)
                
                predictions = expert(latent, training=True)
                
                loss = keras.losses.sparse_categorical_crossentropy(
                    y_batch, predictions
                )
                loss = tf.reduce_mean(loss)
            
            # Only compute gradients for encoder + expert
            trainable_vars = arbiter_trainable_vars + expert.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            
            train_loss.append(loss.numpy())
            acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(predictions, axis=1), 
                                tf.cast(y_batch, tf.int64)), tf.float32)
            )
            train_acc.append(acc.numpy())
        
        # Validation loop
        for x_batch, y_batch in val_data:
            latent, _, _ = arbiter(x_batch, training=False)
            
            predictions = expert(latent, training=False)
            
            loss = keras.losses.sparse_categorical_crossentropy(
                y_batch, predictions
            )
            loss = tf.reduce_mean(loss)
            val_loss.append(loss.numpy())
            
            acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(predictions, axis=1), 
                                tf.cast(y_batch, tf.int64)), tf.float32)
            )
            val_acc.append(acc.numpy())
        
        try:
            import wandb
            wandb.log({
                "epoch": epoch,
                "expert_train_loss": np.mean(train_loss),
                "expert_train_acc": np.mean(train_acc),
                "expert_val_loss": np.mean(val_loss),
                "expert_val_acc": np.mean(val_acc)
            })
        except:
            pass
        
        print(f"Epoch {epoch}: Train Loss={np.mean(train_loss):.4f}, "
              f"Train Acc={np.mean(train_acc):.4f}, "
              f"Val Loss={np.mean(val_loss):.4f}, "
              f"Val Acc={np.mean(val_acc):.4f}")          

def distill_model(student_model, teacher_model, train_data, 
                  temperature=10.0, alpha=0.5, epochs=30):
    """Knowledge distillation for model compression"""
    print("\nPerforming Knowledge Distillation...")
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    
    for epoch in range(epochs):
        epoch_loss = []
        
        for x_batch, y_batch in train_data:
            with tf.GradientTape() as tape:
                # Teacher predictions (soft targets)
                teacher_logits = teacher_model(x_batch, training=False)
                teacher_probs = tf.nn.softmax(teacher_logits / temperature)
                
                # Student predictions
                student_logits = student_model(x_batch, training=True)
                student_probs = tf.nn.softmax(student_logits / temperature)
                
                # Distillation loss
                distill_loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(
                        teacher_probs, student_probs
                    )
                ) * (temperature ** 2)
                
                # Hard target loss
                hard_loss = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(
                        y_batch, student_logits, from_logits=True
                    )
                )
                
                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * hard_loss
            
            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            epoch_loss.append(loss.numpy())
        
        if epoch % 5 == 0:
            print(f"Distillation Epoch {epoch}: Loss={np.mean(epoch_loss):.4f}")