import gym
import lqr
from sklearn.linear_model import Ridge
import autograd.numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def dynamics(x, u):

    th = x[0]*1.0
    thdot = x[1]*1.0

    g = 9.81
    m = 1.
    l = 1.
    dt = 0.05

    u = np.clip(u, -2, 2)[0]

    #print("thdot",thdot.dtype)
    #print("th", th.dtype)
    newthdot = thdot + (-3*g/(2*l) * np.sin(th*1.0 + np.pi) + 3./(m*l**2)*u) * dt
    newth = th + newthdot*dt
    newthdot = np.clip(newthdot, -8, 8)

    x = np.array([newth, newthdot])
    return x

def makeState(X):
    theta = np.arctan2(X[1],X[0])
    if theta >= 0:
        theta = theta - 2*np.pi
    theta = normalize_angle(theta)
    return np.array((theta,X[2]))

def normalize_angle(X):
    return ((X+np.pi)%(2*np.pi)) - np.pi

def cost(X, U):
    return normalize_angle((X[0]**2+1/10*X[1]+1/1000*U**2))

def random_sample(env, iters):
    states = []
    targets = []
    state = env.reset()
    state = makeState(state)
    
    for t in range(iters):
        u = np.random.uniform(-2,2)
        state_next,_,done,_ = env.step([u])
        state_next = makeState(state_next)
        states.append(np.append(state,u))
        target = state_next - state
        targets.append(target)
        state = state_next
        if done:
            break
    return np.array(states), np.array(targets)

horizon = 15

env = gym.make('Pendulum-v1')
env.reset()

alpha = np.linspace(0.01, 0.1, 10)  # good condition
gamma = np.linspace(0.5, 2, 10)
model = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                         param_grid={"alpha": alpha, "gamma": gamma},
                         scoring='neg_mean_squared_error', return_train_score=True)
controller = lqr.iLQR(dynamics, lambda x,u:cost(x,u), 1, 2, horizon)

state,_,_,_ = env.step([0])
state = makeState(state)
u = [np.array((0.0,)) for i in range(horizon)]
states = []
states.append(state)
X,Y = random_sample(env, horizon)
model.fit(X,Y)
costs = []

for i_episode in range(15000):
    x = [state]
    costs.append(cost(x[0], u[0])) 
    for j in range(horizon):
        pred = dynamics(x[j], u[j])
        x.append(pred)
    for j in range(3):
        k, K = controller.backward(x,u)
        x, u = controller.forward(model, x,u ,k , K, 0.99, dynamics)
     
    env.render()
    state_next,r,_,_ = env.step(u[0])
    state_next = makeState(state_next)
    print("reward: ", r, " iter: ", i_episode)
    if i_episode % 10 == 0:
        new_X = np.array([np.hstack((state_next, u[0]))])
        new_Y = np.array([state_next - state])
        X = np.concatenate((X,new_X))
        Y = np.concatenate((Y,new_Y))
        model.fit(X,Y)
    state = state_next

env.close()