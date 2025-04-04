import autograd.numpy as np
from autograd import grad, jacobian
from typing import List, Tuple, Callable, Any
import numpy as np

class iLQR:
    """
    Iterative Linear Quadratic Regulator (iLQR) implementation.
    
    This class implements the iLQR algorithm for trajectory optimization,
    which iteratively linearizes the dynamics and solves for optimal control
    using the Bellman equation.
    """

    def __init__(self, 
                 model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 cost: Callable[[np.ndarray, np.ndarray], float],
                 actionspace_dim: int,
                 statepace_dim: int,
                 iters: int = 100):
        """
        Initialize the iLQR controller.
        
        Args:
            model: Dynamics function that takes state and action as input
            cost: Cost function that takes state and action as input
            actionspace_dim: Dimension of the action space
            statepace_dim: Dimension of the state space
            iters: Number of iterations for the backward pass
        """
        self.itercnt = iters
        self.f = model
        self.a_dim = actionspace_dim
        self.s_dim = statepace_dim
        ##Model hessian
        self.f_x = jacobian(self.f,0)
        self.f_u = jacobian(self.f,1)
        #self.f_xx = jacobian(self.f_x,0)
        #self.f_xu = jacobian(self.f_x,1)
        #self.f_uu = jacobian(self.f_u,1)
        self.C = cost
        self.C_x = grad(self.C, 0)
        self.C_u = grad(self.C, 1)
        self.C_xx = jacobian(self.C_x, 0)
        self.C_ux = jacobian(self.C_u, 0)
        self.C_uu = jacobian(self.C_u, 1)
        ##Initialize value function
        self.V = [0.0 for i in range(iters + 1)]
        self.Vx = [np.zeros(statepace_dim) for i in range(iters + 1)]
        self.Vxx = [np.zeros((statepace_dim,statepace_dim)) for i in range(iters + 1)]

    def backward(self, X: List[np.ndarray], U: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform the backward pass of iLQR.
        
        Args:
            X: List of states along the trajectory
            U: List of controls along the trajectory
            
        Returns:
            Tuple of (k_seq, K_seq) where:
            - k_seq: List of feedforward gains
            - K_seq: List of feedback gains
        """
        ## Set loss function
        self.V[-1] = self.C(X[-1], U[-1])
        self.Vx[-1] = self.C_x(X[-1], U[-1])
        self.Vxx[-1] = self.C_xx(X[-1], U[-1])
    
        k_seq = []
        K_seq = []
    
        for t in range(self.itercnt - 1,-1,-1):
            x = X[t]
            u = U[t]
        
            #update gradient
            fx_t = self.f_x(x,u)
            fu_t = self.f_u(x,u)
            Cx_t = self.C_x(x,u)
            Cu_t = self.C_u(x,u)
            Cxx_t = self.C_xx(x,u)
            Cux_t = self.C_ux(x,u)
            Cuu_t = self.C_uu(x,u)
            ## Compute actionvalue function
            Q_x = Cx_t + fx_t.T@self.Vx[t+1]
            Q_u = Cu_t + fu_t.T@self.Vx[t+1]
            Q_xx = Cxx_t + (fx_t.T@self.Vxx[t+1])@fx_t
            Q_ux  = Cux_t + (fu_t.T@self.Vxx[t+1])@fx_t
            Q_uu = Cuu_t + (fu_t.T@self.Vxx[t+1])@fu_t
            ##Compute an optimal action
            Quu_inv = np.linalg.inv(Q_uu.reshape(1,1) + 1e-09*np.eye(Q_uu.shape[0]))
            k = -Quu_inv@Q_u
            K = -Quu_inv@Q_ux

            ##Update V
            self.Vx[t] = Q_x - K.T@Q_uu@k
            self.Vxx[t] = Q_xx - K.T@Q_uu@K 
            self.V[t] += 0.5*k.T@Q_uu@k + k.T@Q_u
            k_seq.append(k)
            K_seq.append(K)
        K_seq.reverse()
        k_seq.reverse()
        return k_seq, K_seq

    def forward(self, 
                model: Any,
                X: List[np.ndarray],
                U: List[np.ndarray],
                k: List[np.ndarray],
                K: List[np.ndarray],
                alpha: float,
                dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform the forward pass of iLQR.
        
        Args:
            model: Learned dynamics model
            X: List of states along the trajectory
            U: List of controls along the trajectory
            k: List of feedforward gains
            K: List of feedback gains
            alpha: Line search parameter
            dynamics: True dynamics function
            
        Returns:
            Tuple of (x_hat, u_hat) where:
            - x_hat: Updated states
            - u_hat: Updated controls
        """
        x_hat = np.array(X)
        u_hat = np.array(U)
        for t in range(len(U)):
            u = alpha**t*k[t]+K[t]@(x_hat[t]-X[t])
            u_hat[t] = np.clip(U[t] + u, -2,2)
            x_hat[t+1] = model.predict(np.array([np.append(x_hat[t], u_hat[t])])) + x_hat[t]
            #x_hat[t+1] = dynamics(x_hat[t], u_hat[t]) + x_hat[t]
        return x_hat, u_hat 