import gymnasium as gym
import lqr
from sklearn.linear_model import Ridge
import autograd.numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from typing import Tuple, List, Callable, Any
from dataclasses import dataclass

@dataclass
class PendulumConfig:
    """Configuration parameters for the pendulum environment."""
    g: float = 9.81
    m: float = 1.0
    l: float = 1.0
    dt: float = 0.05
    max_force: float = 2.0
    min_force: float = -2.0
    max_velocity: float = 8.0
    min_velocity: float = -8.0

def dynamics(x: np.ndarray, u: np.ndarray, config: PendulumConfig = PendulumConfig()) -> np.ndarray:
    """
    Compute the next state of the pendulum system.
    
    Args:
        x: Current state [theta, theta_dot]
        u: Control input (torque)
        config: Pendulum configuration parameters
        
    Returns:
        Next state [theta, theta_dot]
    """
    th, thdot = x[0], x[1]
    u = np.clip(u, config.min_force, config.max_force)[0]

    newthdot = thdot + (-3*config.g/(2*config.l) * np.sin(th + np.pi) + 
                        3./(config.m*config.l**2)*u) * config.dt
    newth = th + newthdot*config.dt
    newthdot = np.clip(newthdot, config.min_velocity, config.max_velocity)

    return np.array([newth, newthdot])

def makeState(X: np.ndarray) -> np.ndarray:
    """
    Convert cartesian coordinates to pendulum state.
    
    Args:
        X: State in cartesian coordinates [x, y, theta_dot]
        
    Returns:
        State in pendulum coordinates [theta, theta_dot]
    """
    theta = np.arctan2(X[1], X[0])
    if theta >= 0:
        theta = theta - 2*np.pi
    theta = normalize_angle(theta)
    return np.array((theta, X[2]))

def normalize_angle(X: float) -> float:
    """
    Normalize angle to be between -pi and pi using differentiable operations.
    
    This implementation uses trigonometric functions to ensure differentiability
    while maintaining the same behavior as the modulo version.
    
    Args:
        X: Input angle in radians
        
    Returns:
        Normalized angle in radians between -pi and pi
    """
    # Use atan2(sin(x), cos(x)) which is differentiable and gives same result as modulo
    return np.arctan2(np.sin(X), np.cos(X))

def cost(X: np.ndarray, U: np.ndarray) -> float:
    """
    Compute the cost for a given state and control input.
    
    Args:
        X: Current state
        U: Control input
        
    Returns:
        Cost value
    """
    return X[0]**2 + 1/10*X[1] + 1/100*U**2

def random_sample(env: gym.Env, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random samples from the environment.
    
    Args:
        env: Gym environment
        iters: Number of iterations to sample
        
    Returns:
        Tuple of (states, targets) arrays
    """
    states = []
    targets = []
    observation, _ = env.reset()
    state = makeState(observation)
    
    for t in range(iters):
        u = np.random.uniform(-2, 2)
        observation, reward, terminated, truncated, _ = env.step([u])
        state_next = makeState(observation)
        states.append(np.append(state, u))
        target = state_next - state
        targets.append(target)
        state = state_next
        if terminated or truncated:
            break
    return np.array(states), np.array(targets)

def train_model(X: np.ndarray, Y: np.ndarray) -> KernelRidge:
    """
    Train the KernelRidge model with grid search.
    
    Args:
        X: Input features
        Y: Target values
        
    Returns:
        Trained KernelRidge model
    """
    alpha = np.linspace(0.01, 0.1, 10)
    gamma = np.linspace(0.5, 2, 10)
    model = GridSearchCV(
        KernelRidge(kernel='rbf', gamma=0.1),
        cv=5,
        param_grid={"alpha": alpha, "gamma": gamma},
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    model.fit(X, Y)
    return model

def main():
    """Main training loop for the iLQR algorithm."""
    horizon = 30
    

    env = gym.make('Pendulum-v1', render_mode='human')
    # Initial data collection
    X, Y = random_sample(env, horizon)
    model = train_model(X, Y)
    costs = []
    observation, _ = env.reset()

    # Initialize controller and model
    controller = lqr.iLQR(dynamics, cost, 1, 2, horizon)
    
    # Get initial state from environment
    observation, reward, terminated, truncated, _ = env.step([0])
    state = makeState(observation)
    u = [np.array((np.random.uniform(-1, 1),)) for _ in range(horizon)]
    states = [state]

    # Training loop
    for i_episode in range(1500):
        x = [state]
        costs.append(cost(x[0], u[0]))
        
        # Forward pass
        for j in range(horizon):
            pred = dynamics(x[j], u[j])
            x.append(pred)
            
        # Backward pass and policy update
        for j in range(3):
            k, K = controller.backward(x, u)
            x, u = controller.forward(model, x, u, k, K, 0.99, dynamics)
         
        env.render()
        observation, reward, terminated, truncated, _ = env.step(u[0])
        state_next = makeState(observation)
        print(f"reward: {reward}, iter: {i_episode}")
        
        # Update model periodically
        if i_episode % 10 == 0 and i_episode > 5000:
            new_X = np.array([np.hstack((state_next, u[0]))])
            new_Y = np.array([state_next - state])
            X = np.concatenate((X, new_X))
            Y = np.concatenate((Y, new_Y))
            model = train_model(X, Y)
            
        state = state_next

    env.close()

if __name__ == "__main__":
    main()
