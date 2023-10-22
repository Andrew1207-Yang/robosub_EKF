import numpy as np

class EKF:
    """
    Parameters
    ---------------
    dim_x: int
        Dimension of state vector

    dim_z: int
        Dimension of measurement vector  
    
    dim_u: int
        Dimension of control vector

    Attributes
    ---------------
    x: 
        State estimate
    P:
        Covariance
    R:
        Measurement noise
    Q:
        Process noise
    F:
        Jacobian of motion model
    C:
        Jacobian of measurement model
    K:
        Kalman gain
        
    _x_prior:
        Prior(predicted) state
    _x_post:
        Posterior(updated) state
    _P_prior:
        Prior(predicted) covariance
    _P_post:
        Posterior(updated) covariance
    
    """
    def __init__(self, dim_x, dim_z, dim_u):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.C = np.eye(dim_z)
        self.K = np.zeros((dim_x, dim_z))

        self._x_prior = self.x
        self._x_post = self.x
        self._P_prior = self.P
        self._P_post = self.P

    def motion_model(self, x_k, u_k):
        """
        Returns x_{k+1}.
        """

    def measurement_model(self, x_k):
        """
        Return z_{k+1}
        """

    def compute_motion_jacobian(self, x_k, u_k):
        """
        Compute F_k
        """

    def compute_measurement_jacobian(self, x_k):
        """
        Compute C_k
        """
    
    def compute_kalman_gain(self, P_k, C_k, R_k):
        """
        Compute K_k
        """
        first = np.matmul(P_k, np.transpose(C_k))
        CPC = np.matmul(np.matmul(C_k, P_k), np.transpose(C_k)) + R_k
        second = np.linalg.inv(CPC)

        self.K = np.dot(first, second)

    def predict_prior(self, u_k):
        """
        Compute _x_prior with motion_model and _P_prior 
        """
        self._x_prior = self.motion_model(self._x_post, u_k)
        self._P_prior = np.matmul(self.F, np.matmul(self._P_post, np.transpose(self.F))) + self.Q
    
    def update_posterior(self, z_k):
        """
        Compute _x_post and _P_post with Kalman gain
        """
        self._x_post = self._x_prior + np.matmul(self.K, z_k - self.measurement_model(self._x_prior))
        self._P_post = np.matmul((np.eye(self.dim_x) - np.matmul(self.K, self.C)), self._P_prior)
    
    def run(self):
        """
        Run everything
        """
        self.compute_motion_jacobian()
        self.compute_measurement_jacobian()
        self.predict_prior()
        self.compute_kalman_gain()
        self.update_posterior()