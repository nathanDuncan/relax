import numpy as np

config = {
     'num_agents': 5,
     'K' : 5,                                    # Horizon
     'h' : 0.01,                                   # Discretization step
     'kappa' : 1,                                # Steps in the cost function
     'Q' : np.array([100, 100, 0]),            # Position weights

     'X_0' : np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],        # Position of UAVs
                       [0.15, -0.07, 0.0, 0.0, 0.0, 0.0],
                       [0.35, 0.07, 0.0, 0.0, 0.0, 0.0],
                       [0.55, -0.07, 0.0, 0.0, 0.0, 0.0],
                       [0.75, 0.07, 0.0, 0.0, 0.0, 0.0]]),
     'P_DES' : np.array([[0.85, 0.0, 0.0],               # 1 ego agent, 3 state components, (x y z)
                         [0.15, -0.07, 0.0],               # Static Agent
                         [0.35, 0.07, 0.0],
                         [0.55, -0.07, 0.0],
                         [0.75, 0.07, 0.0]]),             # Static Agent

      
     'r_min': 0.1,                            
     'theta': np.array([1, 1, 2]),
     'lambda_reg': 0.01,
}

# config = {
#      'num_agents': 2,
#      'K' : 1,                                    # Horizon
#      'h' : 0.05,                                   # Discretization step
#      'kappa' : 1,                                # Steps in the cost function
#      'Q' : np.array([500, 250, 0]),            # Position weights

#      'X_0' : np.array([[0, 0, 0, 0, 0, 0],        # Position of UAVs
#                        [0.1, 0., 0, 0, 0, 0]]),
#      'P_DES' : np.array([[0.5, 0.05, 0],               # 1 ego agent, 3 state components, (x y z)
#                          [0.1, 0., 0]]),             # 1 static agent

      
#      'r_min': 0.1,                            
#      'theta': np.array([1, 1, 2])
# }