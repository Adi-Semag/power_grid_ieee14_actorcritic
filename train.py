import numpy as np
import torch
import matplotlib.pyplot as plt
from power_grid_env import PowerGridEnv
from actor_critic import ActorCritic
import time
import os
from datetime import datetime

def train(env, agent, num_episodes=3000, max_steps=300, eval_interval=10, 
          eval_episodes=10, save_interval=100, patience=100, improvement_threshold=0.05):
    """
    Train the agent with improved monitoring and curriculum learning
    """
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize metrics
    episode_rewards = []
    eval_rewards = []
    actor_losses = []
    critic_losses = []
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    curriculum_stage = 0
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        steps_without_improvement = 0
        
        for step in range(max_steps):
            # Select action with exploration
            action = agent.select_action(state, training=True)
            
            # Take action and observe result
            next_state, reward, done, info = env.step(action)
            
            # Store transition and update agent
            actor_loss, critic_loss = agent.update(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            if actor_loss is not None:
                episode_actor_loss += actor_loss
            if critic_loss is not None:
                episode_critic_loss += critic_loss
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        if episode_actor_loss > 0:
            actor_losses.append(episode_actor_loss / (step + 1))
        if episode_critic_loss > 0:
            critic_losses.append(episode_critic_loss / (step + 1))
        
        # Evaluate agent periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent, eval_episodes)
            eval_rewards.append(eval_reward)
            
            # Check for improvement
            if eval_reward > best_eval_reward + improvement_threshold:
                best_eval_reward = eval_reward
                no_improvement_count = 0
                # Save best model
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'optimizer_actor_state_dict': agent.actor_optimizer.state_dict(),
                    'optimizer_critic_state_dict': agent.critic_optimizer.state_dict(),
                    'episode': episode,
                    'eval_reward': eval_reward,
                }, os.path.join(results_dir, 'best_model.pth'))
            else:
                no_improvement_count += 1
            
            # Check for curriculum stage advancement
            if agent.update_curriculum_stage(eval_reward):
                curriculum_stage += 1
                print(f"Advanced to curriculum stage {curriculum_stage}")
            
            # Early stopping check
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered after {episode + 1} episodes")
                break
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Average Reward: {np.mean(episode_rewards[-eval_interval:]):.2f}")
            print(f"Evaluation Reward: {eval_reward:.2f}")
            print(f"Best Evaluation Reward: {best_eval_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Curriculum Stage: {curriculum_stage}")
            if actor_losses:
                print(f"Actor Loss: {np.mean(actor_losses[-eval_interval:]):.6f}")
            if critic_losses:
                print(f"Critic Loss: {np.mean(critic_losses[-eval_interval:]):.6f}")
        
        # Save checkpoint periodically
        if (episode + 1) % save_interval == 0:
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_actor_state_dict': agent.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': agent.critic_optimizer.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards,
                'eval_rewards': eval_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
            }, os.path.join(results_dir, f'checkpoint_{episode + 1}.pth'))
    
    # Plot and save results
    plot_results(episode_rewards, eval_rewards, actor_losses, critic_losses, results_dir)
    
    return episode_rewards, eval_rewards, actor_losses, critic_losses

def evaluate(env, agent, num_episodes=10):
    """
    Evaluate the agent without exploration
    """
    eval_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)

def plot_results(episode_rewards, eval_rewards, actor_losses, critic_losses, results_dir):
    """
    Plot and save training results
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    plt.plot(eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    
    # Plot actor losses
    plt.subplot(2, 2, 3)
    plt.plot(actor_losses)
    plt.title('Actor Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot critic losses
    plt.subplot(2, 2, 4)
    plt.plot(critic_losses)
    plt.title('Critic Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_results.png'))
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment and agent
    env = PowerGridEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr_actor=5e-5,
        lr_critic=1e-4,
        gamma=0.99,
        tau=0.001,
        buffer_size=200000,
        batch_size=128,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train agent
    episode_rewards, eval_rewards, actor_losses, critic_losses = train(
        env=env,
        agent=agent,
        num_episodes=3000,
        max_steps=300,
        eval_interval=10,
        eval_episodes=10,
        save_interval=100,
        patience=100,
        improvement_threshold=0.05
    )
    
    # Final evaluation
    final_eval_reward = evaluate(env, agent, num_episodes=20)
    print(f"\nFinal Evaluation Results:")
    print(f"Average Reward: {final_eval_reward:.2f}")
    
    # Plot and save results
    plot_results(episode_rewards, eval_rewards, actor_losses, critic_losses, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}") 