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

def get_expert_for_label(label, expert_labels):
    for expert_id, labels in expert_labels.items():
        if label in labels:
            return expert_id
    return None
    
def label_to_expert_label(label, expert_id, expert_labels):
    expert_labels = expert_labels[expert_id]
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
                    expert_labels,
                    episodes=1000,
                    algorithm='PPO',
                    trajectory_length=128):  # Larger batches for GPU efficiency
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
        episode_total_reward = 0
        episode_batches = 0
        
        # Keep everything on GPU - use lists of tensors
        trajectory_rewards = []
        trajectory_actions = []
        trajectory_old_probs = []
        trajectory_inputs = []
        trajectory_labels = []
        
        batch_count = 0
        
        # === EXPERIENCE COLLECTION ===
        for x_batch, y_batch in train_data:
            batch_size = tf.shape(x_batch)[0]
            
            # Forward pass (keep on GPU)
            latent, policy_logits, values = arbiter(x_batch, training=False)
            
            actions = sample_action(policy_logits)
            actions = tf.squeeze(actions, axis=1)
            
            # Store old action probabilities
            action_probs = tf.nn.softmax(policy_logits)
            action_probs = tf.cast(action_probs, tf.float32)

            action_masks = tf.one_hot(actions, NUM_ACTIONS)

            old_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            
            # Calculate rewards (CPU loop but minimal)
            rewards_list = []
            actions_np = actions.numpy()
            y_batch_np = y_batch.numpy()
            latent_np = latent.numpy()
            
            for i in range(batch_size):
                action = actions_np[i]
                true_label = y_batch_np[i]
                
                if action == 0:
                    reward = NO_DETECTION_REWARD
                else:
                    expert_id = action - 1
                    correct_expert = get_expert_for_label(true_label, expert_labels)
                    
                    if expert_id == correct_expert:
                        expert_latent = tf.expand_dims(latent[i:i+1], 0)
                        pred = experts[expert_id](expert_latent, training=False)
                        expert_label = label_to_expert_label(true_label, expert_id, expert_labels)
                        
                        if tf.argmax(pred[0]).numpy() == expert_label:
                            reward = CORRECT_REWARD + EFFICIENCY_BONUS
                        else:
                            reward = INCORRECT_PENALTY
                    else:
                        reward = INCORRECT_PENALTY
                
                rewards_list.append(reward)
            
            rewards = tf.constant(rewards_list, dtype=tf.float32)
            
            # Store trajectory data (KEEP ON GPU - just store references)
            trajectory_rewards.append(rewards)
            trajectory_actions.append(actions)
            trajectory_old_probs.append(old_probs)
            trajectory_inputs.append(x_batch)
            trajectory_labels.append(y_batch)
            
            episode_total_reward += tf.reduce_sum(rewards).numpy()
            episode_batches += 1
            batch_count += 1
            
            # Don't delete tensors yet - we need them
            
            # === PERFORM PPO UPDATE EVERY trajectory_length BATCHES ===
            if batch_count >= trajectory_length:
                if algorithm == 'PPO':
                    # Concatenate all trajectory batches into single tensors (more efficient)
                    all_rewards = tf.concat(trajectory_rewards, axis=0)
                    all_actions = tf.concat(trajectory_actions, axis=0)
                    all_old_probs = tf.concat(trajectory_old_probs, axis=0)
                    all_inputs = tf.concat(trajectory_inputs, axis=0)
                    
                    # Compute returns once (outside PPO epochs)
                    all_returns = compute_discounted_returns(all_rewards, gamma=0.99)
                    
                    # Create a tf.data.Dataset for efficient batching during PPO epochs
                    update_dataset = tf.data.Dataset.from_tensor_slices({
                        'inputs': all_inputs,
                        'actions': all_actions,
                        'old_probs': all_old_probs,
                        'returns': all_returns
                    })
                    update_dataset = update_dataset.shuffle(buffer_size=len(all_rewards))
                    update_dataset = update_dataset.batch(64)  # Mini-batch for updates
                    
                    for ppo_epoch in range(ppo_epochs):
                        epoch_policy_loss = 0
                        epoch_value_loss = 0
                        epoch_entropy = 0
                        n_batches = 0
                        
                        for mini_batch in update_dataset:
                            with tf.GradientTape() as tape:
                                # Forward pass
                                _, new_policy, new_values = arbiter(mini_batch['inputs'], training=True)

                                new_policy = tf.cast(new_policy, tf.float32)
                                new_values = tf.cast(new_values, tf.float32)

                                new_probs_all = tf.nn.softmax(new_policy)
                                
                                # Compute advantages
                                advantages = mini_batch['returns'] - tf.squeeze(new_values)
                                advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
                                
                                # PPO clipped objective
                                action_masks = tf.one_hot(mini_batch['actions'], NUM_ACTIONS)
                                new_probs = tf.reduce_sum(new_probs_all * action_masks, axis=1)
                                
                                ratio = new_probs / (mini_batch['old_probs'] + 1e-10)
                                surr1 = ratio * advantages
                                surr2 = tf.clip_by_value(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
                                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                                
                                # Value loss
                                value_loss = tf.reduce_mean(tf.square(mini_batch['returns'] - tf.squeeze(new_values)))
                                
                                # Entropy bonus
                                entropy = -tf.reduce_mean(
                                    tf.reduce_sum(new_probs_all * tf.math.log(new_probs_all + 1e-10), axis=1)
                                )

                                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                            
                            # Apply gradients
                            grads = tape.gradient(total_loss, arbiter.trainable_variables)
                            grads, _ = tf.clip_by_global_norm(grads, 0.5)
                            optimizer.apply_gradients(zip(grads, arbiter.trainable_variables))
                            
                            epoch_policy_loss += policy_loss
                            epoch_value_loss += value_loss
                            epoch_entropy += entropy
                            n_batches += 1
                        
                        # Log after each PPO epoch
                        if ppo_epoch == ppo_epochs - 1:
                            avg_reward = episode_total_reward / episode_batches
                            print(f"Episode {episode}, Batch {episode_batches}: Avg Reward={avg_reward:.4f}, "
                                  f"Loss={total_loss.numpy():.4f}")
                            
                            wandb.log({
                                "episode": episode,
                                "batch": episode_batches,
                                "avg_reward": avg_reward,
                                "policy_loss": (epoch_policy_loss / n_batches).numpy(),
                                "value_loss": (epoch_value_loss / n_batches).numpy(),
                                "entropy": (epoch_entropy / n_batches).numpy(),
                                "total_loss": total_loss.numpy()
                            })
                    
                    # Clear trajectory after update
                    del all_rewards, all_actions, all_old_probs, all_inputs, all_returns, update_dataset
                
                # Clear trajectory buffers
                trajectory_rewards = []
                trajectory_actions = []
                trajectory_old_probs = []
                trajectory_inputs = []
                trajectory_labels = []
                batch_count = 0
                
                # Only clear session periodically, not every update
                if episode_batches % 10 == 0:
                    tf.keras.backend.clear_session()

        arbiter.save_weights(f'./arbiter.weights.h5')
        
        # Episode summary
        if episode % 10 == 0:
            avg_episode_reward = episode_total_reward / episode_batches
            print(f"\n=== Episode {episode} Complete: Avg Reward={avg_episode_reward:.4f} ===\n")
            
def train_expert_supervised(arbiter, expert, expert_id, train_data, val_data, epochs=100):
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

        expert.load_weights(f'./expert_{expert_id}.weights.h5')
        
        print(f"Epoch {epoch}: Train Loss={np.mean(train_loss):.4f}, "
              f"Train Acc={np.mean(train_acc):.4f}, "
              f"Val Loss={np.mean(val_loss):.4f}, "
              f"Val Acc={np.mean(val_acc):.4f}")          


def train_arbiter_distillation_with_features(student_arbiter,
                                             teacher_arbiter,
                                             train_data,
                                             epochs=2,
                                             temperature=5.0,
                                             alpha=0.7,
                                             beta=0.3,
                                             learning_rate=0.0001):
    
    print(f"\nTraining Student Arbiter with Feature Distillation...")
    print(f"Temperature: {temperature}, Alpha: {alpha}, Beta (feature): {beta}")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        epoch_distill_loss = 0
        epoch_hard_loss = 0
        epoch_feature_loss = 0
        epoch_total_loss = 0
        epoch_policy_acc = 0
        n_batches = 0
        
        for x_batch, y_batch in train_data:            
            with tf.device('/GPU:0'):
                teacher_latent, teacher_policy_logits, teacher_values = teacher_arbiter(
                    x_batch, training=False
                )
                teacher_probs_soft = tf.nn.softmax(teacher_policy_logits / temperature)
                teacher_probs_hard = tf.nn.softmax(teacher_policy_logits)
            
            with tf.GradientTape() as tape:
                student_latent, student_policy_logits, student_values = student_arbiter(
                    x_batch, training=True
                )
                
                student_probs_soft = tf.nn.softmax(student_policy_logits / temperature)
                student_probs_hard = tf.nn.softmax(student_policy_logits)
                
                distill_loss = -tf.reduce_mean(
                    tf.reduce_sum(teacher_probs_soft * tf.math.log(student_probs_soft + 1e-10), axis=1)
                ) * (temperature ** 2)
                
                hard_loss = -tf.reduce_mean(
                    tf.reduce_sum(teacher_probs_hard * tf.math.log(student_probs_hard + 1e-10), axis=1)
                )
                
                value_loss = tf.reduce_mean(tf.square(teacher_values - student_values))
                
                # feature_loss = tf.reduce_mean(tf.square(teacher_latent - student_latent))
                
                total_loss = (alpha * distill_loss + 
                             (1 - alpha) * hard_loss + 
                             0.5 * value_loss )
            
            grads = tape.gradient(total_loss, student_arbiter.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, student_arbiter.trainable_variables))
            
            student_actions = tf.argmax(student_probs_hard, axis=1)
            teacher_actions = tf.argmax(teacher_probs_hard, axis=1)
            policy_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(student_actions, teacher_actions), tf.float32)
            )
            
            epoch_distill_loss += distill_loss.numpy()
            epoch_hard_loss += hard_loss.numpy()
            # epoch_feature_loss += feature_loss.numpy()
            epoch_total_loss += total_loss.numpy()
            epoch_policy_acc += policy_accuracy.numpy()
            n_batches += 1
        
        if epoch % 1 == 0:
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Distill Loss: {epoch_distill_loss/n_batches:.4f}")
            print(f"  Hard Loss: {epoch_hard_loss/n_batches:.4f}")
            # print(f"  Feature Loss: {epoch_feature_loss/n_batches:.4f}")
            print(f"  Total Loss: {epoch_total_loss/n_batches:.4f}")
            print(f"  Policy Accuracy: {epoch_policy_acc/n_batches:.4f}")
            
            wandb.log({
                "distill_epoch": epoch,
                "distill_loss": epoch_distill_loss/n_batches,
                "hard_loss": epoch_hard_loss/n_batches,
                # "feature_loss": epoch_feature_loss/n_batches,
                "total_loss": epoch_total_loss/n_batches,
                "policy_accuracy": epoch_policy_acc/n_batches
            })
    
    print("\n=== Feature Distillation Complete ===")
    return student_arbiter