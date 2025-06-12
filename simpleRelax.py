"""
A simple test of SDP relaxation collision avoidance.
"""
# Imports
import os
import importlib
import numpy as np
import mosek.fusion as mf
from qpsolvers import solve_qp

# Plots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter

# Global
terminal_width = os.get_terminal_size().columns

class Agent():
    """
    Simple class/object to hold all the relevant info for an agent.
    """
    def __init__(self, ID, config):
        self.ID = ID
        self._K = config['K']
        self._h = config['h']
        self._kappa = config['kappa']
        self.r_min = config['r_min']          # Physical Radius
        a = 1
        self.neighbourhood = a * self.r_min     # Collision boundary for search
        self.THETA = np.diag(config['theta']) # Ellipsoid Axes Scale
        self.THETA_INV = np.linalg.inv(self.THETA)
        self.THETA_INV2 = np.linalg.matrix_power(self.THETA_INV, 2)

        self.X_0 = config['X_0'][ID]
        self.P_DES = config['P_DES'][ID]

        self.acc_max = 10.0 #m/s^2

        self.OMEGA = np.empty([0])            # List of potential colliding agents
        self.kc = -np.inf                     # Earliest predicted collision instance

        ##### MODEL ###########################################################################
        self.A, self.B = get_agent_models(self._h)
        self.A_0, self.LAMBDA = get_model_prediction_matrices(self._K, self.A, self.B)

        ##### WEIGHTS #########################################################################
        ### Weight ###
        self.Q = np.diag([100, 100, 100])
        self.Q_TILDE = np.diag([])
        for i in range(self._K):
            if i >= self._K-self._kappa:
                self.Q_TILDE = np.block([
                    [self.Q_TILDE, np.zeros((self.Q_TILDE.shape[0], self.Q.shape[1]))],
                    [np.zeros((self.Q.shape[0], self.Q_TILDE.shape[1])), self.Q]
                    ])
            else:
                self.Q_TILDE = np.block([
                    [self.Q_TILDE, np.zeros((self.Q_TILDE.shape[0], self.Q.shape[1]))],
                    [np.zeros((self.Q.shape[0], self.Q_TILDE.shape[1])), np.zeros((self.Q.shape[0], self.Q.shape[1]))]
                    ])
                
        ##### OBJECTIVE #######################################################################
        self.P = None
        self.q = None

        ##### CONSTRAINTS #####################################################################
        ### Continuity (Equality)
        self.A_eq = None
        self.b_eq = None

        ### Input Limitations (Inequality)
        # Acceleration
        self.acc_min = - self.acc_max
        b_acc_max = np.array([self.acc_max for _ in range(3*self._K)])
        A_acc_max = np.diag([1 for _ in range(3*self._K)])
        b_acc_min = -np.array([self.acc_min for _ in range(3*self._K)])
        A_acc_min = np.diag([-1 for _ in range(3*self._K)])
        b_acc = np.hstack((b_acc_max, b_acc_min))
        A_acc = np.vstack((A_acc_max, A_acc_min))

        self.b_in = b_acc
        self.A_in = A_acc

        ### Elipsoid
        self.S_i = None

        self.A_eps = None
        self.b_eps = None
        self.c_eps = None
    
    def step(self, U, dt):
        A, B = get_agent_models(dt)
        X_next = A @ self.X_0 + B @ U
        self.X_0 = X_next
        print(f"Next Position: {X_next}")

        return

################################################################################
def get_agent_models(dt):
    # State Transition Matrix (_s_dim x _s_dim)
    A = np.array([[1, 0, 0, dt,  0,  0],     # px
                  [0, 1, 0,  0, dt,  0],     # py
                  [0, 0, 1,  0,  0, dt],     # pz
                  [0, 0, 0,  1,  0,  0],     # vx
                  [0, 0, 0,  0,  1,  0],     # vy
                  [0, 0, 0,  0,  0,  1]])    # vz

    # Control Input Matrix (_u_dim x _s_dim)
    B = np.array([[ dt**2/2,       0,       0],  # px
                  [       0, dt**2/2,       0],  # py
                  [       0,       0, dt**2/2],  # pz
                  [      dt,       0,       0],  # vx
                  [       0,      dt,       0],  # vy
                  [       0,       0,      dt]]) # vz
    return A, B

################################################################################
def get_model_prediction_matrices(K, A, B):
    """
    (Luis, 2020) II. Problem Statement, B. The Agent Prediction Model
    Equation 4 
    Retruns: A_0 (Stacked State Transition Matrix [3Nx6]), Lambda (Stacked Input Transition Matrix [3Nx3N])
    """

    ##### Constants ############################################################
    Z_3x3 = np.zeros((3, 3))
    I_3x3 = np.eye(3)

    ##### Build PSI for extracting (x, y, z) ###################################
    PSI = np.hstack((I_3x3, Z_3x3))

    ##### Build LAMBDA Matrix (Stacked Input Transition Matrix) ################
    # Equation 9 (Luis, 2018)
    rows = []
    for i in range(K):
        row = []
        # Builds lower triangular elements
        for j in range(i, -1, -1):
            row.append(PSI @ np.linalg.matrix_power(A, j) @ B)
        # Append zero matricies for the remaining columns
        while len(row) < K:
            row.append(Z_3x3)
        rows.append(np.hstack(row))

    LAMBDA = np.vstack(rows)

    ##### Build A0 (Stacked State Transition Matrix) ###########################
    # Equation 10 (Luis, 2018)
    A0_elements = [(PSI @ np.linalg.matrix_power(A, i+1)).T for i in range(K)]
    A0 = np.hstack(A0_elements).T

    return A0, LAMBDA

################################################################################
def load_config_module(path):
    """Dynamically import a config module and return its config dict."""
    print(f"Path: {path}")
    try:
        module = importlib.import_module(path)
        return getattr(module, 'config', {})  # Expect each config module to have `config` dict
    except Exception as e:
        print(f"\n[ERROR] loading parameters from '{path}': {e}\n")
        exit()

################################################################################
def simplest_setup(ego_agent):
    # State: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
    # Input: acc_x, acc_y, acc_z

    # These are really the only two parameters that should be changing
    X_0 = ego_agent.X_0        # Position of UAVs
    P_DES = ego_agent.P_DES    # Desired position, 3 state components, (x y z)
    ### Stacked Goal ###
    P_des = np.tile(P_DES, ego_agent._K)

    ##### OBJECTIVE FUNCTION ##############################################################
    # # # Trajectory Error Penalty # # # # # # # # # # # # # # #
    # J_error = U.T(self.LAMBDA.T @ Q_tilde @ self.LAMBDA)U - 2*(P_d.T @ Q_tilde @ self.LAMBDA - (self.A0 @ X_0).T @ Q_tilde @ self.LAMBDA)U

    ### Terms ###        
    P_error = ego_agent.LAMBDA.T @ ego_agent.Q_TILDE @ ego_agent.LAMBDA
    q_error = -2*(P_des.T @ ego_agent.Q_TILDE @ ego_agent.LAMBDA - (ego_agent.A_0 @ X_0).T @ ego_agent.Q_TILDE @ ego_agent.LAMBDA)

    # # # Cumulative Objective # # # # # # # # # # # # # # # # #
    ego_agent.P = P_error
    ego_agent.q = q_error

    return 

################################################################################
def avoidance_setup(ego_agent, PI):
    
    # Update P and q terms of objective
    simplest_setup(ego_agent)

    # Reset constraints
    ego_agent.A_eps, ego_agent.b_eps, ego_agent.c_eps = None, None, None

    # Predict collisions
    if ego_agent._K == 1:
        ego_agent.kc = 0
        get_quadratic_collision_constraints(ego_agent, PI[int(1), ego_agent.kc])
    else:
        predict_collision(ego_agent, PI)
        # If there are collisions, create constraints
        for j in ego_agent.OMEGA: 
            get_quadratic_collision_constraints(ego_agent, PI[int(j), ego_agent.kc])
        
    return

################################################################################
def predict_collision(ego_agent, PI):
    """
    Determines the earliest time step in which a collision is likely and which agents
    are at risk using PI
    """
    num_agents = PI.shape[0]
    K = ego_agent._K

    OMEGA = np.empty([0])            # List of potential colliding agents
    kc = -np.inf                     # Earliest predicted collision instance

    ### Search for Agents within collision boundary ##################
    for k in range(K):
        for j in range(num_agents):
            if xiEllipsoid(PI[ego_agent.ID, k, :], PI[j, k, :], ego_agent.THETA_INV) < ego_agent.neighbourhood and j != ego_agent.ID:
                print(f"{xiEllipsoid(PI[ego_agent.ID, k, :], PI[j, k, :], ego_agent.THETA_INV)} < {ego_agent.neighbourhood}")
                # A collision is predicted
                print("COLLISION!!!!\nAgent " + str(ego_agent.ID) + " at position", PI[ego_agent.ID, k, :], 
                              " and \nAgent" + str(j) + " at position", PI[j, k, :])
                print(f"\nXi^2 = {(xiEllipsoid(PI[ego_agent.ID, k, :], PI[j, k, :], ego_agent.THETA_INV))**2}")
                print(f"p_i = {PI[ego_agent.ID, k, :]}, p_j = {PI[j, k, :]}")
                if len(OMEGA) == 0:
                    # This is the first collision detected with an agent
                    kc = k - 1
                    #Edge case
                    if kc == -1: kc == 0
                    if k == 0:
                        # Drone is already within the collision boundary
                        kc = 0
                if j not in OMEGA:
                    OMEGA = np.hstack((OMEGA, int(j)))

    ego_agent.kc = kc
    ego_agent.OMEGA = OMEGA
    return

################################################################################
def xiEllipsoid(p_i, p_j, THETA_INV):
    """
    Second order ellipsoid as seen in Equation 8 (Luis, 2020)
    """
    return np.linalg.norm((THETA_INV)@(p_i - p_j), 2)

################################################################################
def get_quadratic_collision_constraints(ego_agent, p_j):
    """
    u^T A_eps u + 2 b_eps u + c_eps
    """

    selector = np.zeros((ego_agent._K*3, 3))
    selector[3*ego_agent.kc, 0] = 1
    selector[3*ego_agent.kc+1, 1] = 1
    selector[3*ego_agent.kc+2, 2] = 1
    ego_agent.S_i = selector.T

    ego_agent.A_eps = ego_agent.LAMBDA.T @ ego_agent.S_i.T @ ego_agent.THETA_INV2 @ ego_agent.S_i @ ego_agent.LAMBDA
    ego_agent.b_eps = (ego_agent.S_i @ ego_agent.A_0 @ ego_agent.X_0 - p_j).T @ ego_agent.THETA_INV2 @ ego_agent.S_i @ ego_agent.LAMBDA
    ego_agent.c_eps = (ego_agent.S_i @ ego_agent.A_0 @ ego_agent.X_0 - p_j).T @ ego_agent.THETA_INV2 @ (ego_agent.S_i @ ego_agent.A_0 @ ego_agent.X_0 - p_j) - ego_agent.r_min**2

    return

################################################################################
def solve_sdp(agent):
    # Problem Dimensions
    n = 3*agent._K

    A_in = agent.A_in
    b_in = agent.b_in
    A_eq = agent.A_eq
    b_eq = agent.b_eq

    A_eps = agent.A_eps
    b_eps = agent.b_eps
    c_eps = agent.c_eps

    lambda_reg = 0.005
    
    # Create the model
    with mf.Model("SDP Relaxation with Ellipsoid") as M:
        ### Decision variables
        U = M.variable("U", [n, n], mf.Domain.inPSDCone())  # PSD matrix U
        u = M.variable("u", n, mf.Domain.unbounded())       # Vector u

        ### Objective: minimize 0.5 * Tr(P U) + q^T u
        obj_no_reg = mf.Expr.add(mf.Expr.mul(0.5, mf.Expr.dot(agent.P, U)), mf.Expr.dot(agent.q.T, u))
        trace_U = mf.Expr.add([U.index(i, i) for i in range(n)])
        reg_term = mf.Expr.mul(lambda_reg, trace_U)

        obj = mf.Expr.add(obj_no_reg, reg_term)
        M.objective("Minimize", mf.ObjectiveSense.Minimize, obj)

        ### Equality constraint
        if A_eq is not None and b_eq is not None:
            M.constraint("Aeq_u_eq_beq", mf.Expr.sub(mf.Expr.mul(A_eq, u), b_eq), mf.Domain.equalsTo(0.0))

        ### Inequality constraint
        M.constraint("Ain_u_le_bin", mf.Expr.sub(mf.Expr.mul(A_in, u), b_in), mf.Domain.lessThan(0.0))

        ### Ellipsoid-based safety constraint:
        # Tr(A_eps * U) + 2 b_eps^T u + c_eps >= 0
        if A_eps is not None:
            ellipsoid_lhs = mf.Expr.add(
                mf.Expr.add(mf.Expr.dot(A_eps.T, U), mf.Expr.mul(2.0, mf.Expr.dot(b_eps.T, u))),
                c_eps
            )
            M.constraint("ellipsoid_safety", ellipsoid_lhs, mf.Domain.greaterThan(0.0))

        ### Schur complement matrix
        Z = M.variable("Z", [n + 1, n + 1], mf.Domain.inPSDCone())
        # Z[:n, :n] == U
        for i in range(n):
            for j in range(n):
                M.constraint(f"Z_U_match_{i}_{j}", mf.Expr.sub(Z.index(i, j), U.index(i, j)), mf.Domain.equalsTo(0.0))
        # Z[:n, n] == u
        for i in range(n):
            M.constraint(f"Z_u_match_{i}", mf.Expr.sub(Z.index(i, n), u.index(i)), mf.Domain.equalsTo(0.0))
        # Z[n, :n] == u^T
        for j in range(n):
            M.constraint(f"Z_uT_match_{j}", mf.Expr.sub(Z.index(n, j), u.index(j)), mf.Domain.equalsTo(0.0))
        # Z[n, n] == 1
        M.constraint("Z_block4", Z.index(n, n), mf.Domain.equalsTo(1.0))

        ### Solve the problem
        M.solve()

        status = M.getProblemStatus(mf.SolutionType.Default)

        if status in [mf.ProblemStatus.PrimalFeasible, mf.ProblemStatus.PrimalAndDualFeasible]:
            try:
                U_val = U.level()
                U_val = U_val.reshape((3*agent._K, 3*agent._K))
                u_val = u.level()
                print("\n\n\n")
                print(f"u:\n{u_val}")
                print(f"uu^T:\n{u_val*u_val.T}")
                print(f"U:\n{U_val}")
                rank = np.linalg.matrix_rank(U_val)
                print(f"rank: {rank}")
                print("\n\n\n")

                # Reconstruct rank-1 matrix from u
                u_outer = np.outer(u_val, u_val)
                print(f"u_outer:\n{u_outer}")

                # Compute the residual
                residual = U_val - u_outer

                # Frobenius norm of the difference
                violation = np.linalg.norm(residual, ord='fro')

                # Optional: relative error
                relative_error = violation / np.linalg.norm(U_val, ord='fro')

                print("||U - uuᵀ||_F =", violation)
                print("Relative error:", relative_error)

                print("\n\n\n")
                Z = np.block([
                            [U_val,          u_val.reshape(-1, 1)],
                            [u_val.reshape(1, -1), np.array([[1.0]])]
                        ])
                is_symmetric = np.allclose(Z, Z.T, atol=1e-8)
                print("Z is symmetric:", is_symmetric)
                eigvals = np.linalg.eigvalsh(Z)  # Use eigvalsh for symmetric matrices
                is_psd = np.all(eigvals >= -1e-8)  # tolerate small negative values due to numerical error

                print("Eigenvalues of Z:", eigvals)
                print("Z is positive semidefinite:", is_psd)

                return u_val, U_val
            except mf.SolutionError:
                print("u is not accessible even though problem status is", status)
                exit()
        elif status == mf.ProblemStatus.Unknown:
            try:
                u_val = u.level(mf.SolutionType.Interior)
                print("Interior solution u:", u_val)
                return u_val
            except:
                print("Interior solution not available.")
                exit()
        else:
            print("No usable solution. Status =", status)
            exit()

################################################################################
def main():
    np.set_printoptions(threshold=np.inf)
    ##### LOAD CONFIGS ###################################################################
    config_path = "simpleRelaxConfig"
    try:
        config = load_config_module(config_path)
    except:
        print(f"[ERROR] The specified parameter file {config_path} could not be found in the current directory.")
        exit()

    agent = [None] * config['num_agents']
    for i in range(config['num_agents']):
        agent[i] = Agent(i, config)

    PI = np.zeros((config['num_agents'], config['K'], 3)) 

    simplest_setup(agent[0])

    ##### SOLVE PROBLEM ###################################################################
    # Firstly use QP to generate an initial prediction
    U_QP = solve_qp(2*agent[0].P, agent[0].q, G=agent[0].A_in, h=agent[0].b_in, solver='osqp', verbose=False)
    # Print Solution
    print(f"Solution QP:")
    for i in range(0, len(U_QP), 3):
        print(*U_QP[i:i+3])
    # Compute State Horizon
    POS_QP = agent[0].A_0 @ agent[0].X_0 + agent[0].LAMBDA @ U_QP

    ### Add Solution to Shared List
    for i in range(config['K']):
        PI[0, i, :] = POS_QP[3*i: 3*(i+1)] # Ego Agent
        PI[1, i, :] = agent[1].X_0[:3]     # Static agent
    print(f"\nPI: {PI}")
    # Traj is for plotting
    TRAJ = np.zeros((config['num_agents'], config['K']+1, 3))
    for k in range(config['K']+1):
        for j in range(config['num_agents']):
            if k == 0:
                TRAJ[j, k, :] = agent[j].X_0[0:3]
            else:
                TRAJ[j, k, :] = PI[j, k-1, :]

    plot_horizon(0, agent, TRAJ)
    agent0_history = agent[0].X_0[0:3]

    ##### MAIN LOOP #####
    for i in range(100):
    
        ### Setup Problem
        avoidance_setup(agent[0], PI)

        ### Solve with obstacles
        u_SDP, U_SDP = solve_sdp(agent[0])
        print("\n\n\n")
        print(f"Solution SDP:")
        for i in range(0, len(u_SDP), 3):
            print(*u_SDP[i:i+3])
        # dist = U_QP @ agent[0].A_eps @ U_QP + 2 * agent[0].b_eps @ U_QP + agent[0].c_eps
        # print(f"dist:\n{dist}")
        # dist_u = u_SDP @ agent[0].A_eps @ u_SDP + 2 * agent[0].b_eps @ u_SDP + agent[0].c_eps
        # print(f"dist_u:\n{dist_u}")
        # dist_U = np.sum(agent[0].A_eps * U_SDP) + 2 * agent[0].b_eps @ u_SDP + agent[0].c_eps
        # print(f"dist_U:\n{dist_U}")

        # Compute State Horizon
        POS_SDP = agent[0].A_0 @ agent[0].X_0 + agent[0].LAMBDA @ u_SDP

        ### Add Solution to Shared List
        for i in range(config['K']):
            PI[0, i, :] = POS_SDP[3*i: 3*(i+1)] # Ego Agent
            PI[1, i, :] = agent[1].X_0[:3]      # Static agent
        print(f"\nPI: {PI}")
        # Traj is for plotting
        TRAJ = np.zeros((config['num_agents'], config['K']+1, 3))
        for k in range(config['K']+1):
            for j in range(config['num_agents']):
                if k == 0:
                    TRAJ[j, k, :] = agent[j].X_0[0:3]
                else:
                    TRAJ[j, k, :] = PI[j, k-1, :]

        plot_horizon(0, agent, TRAJ)

        ### Update Initial Conditions / Step environment
        agent[0].step(u_SDP[:3], dt=0.02)
        agent0_history = np.vstack((agent0_history, agent[0].X_0[0:3]))
        print(f"otherPosition: {TRAJ[0, 1, :]}")

    # play_horizon(agent0_history, agent)

    return 0

#################################################################################
def plot_horizon(ID, agents, TRAJ):
            ### Extract Values ############
            # Terminal Header
            output_title = " Horizon Plot "
            print("\n\n")
            print('#' * 2*terminal_width)
            print('#'*int((terminal_width-len(output_title))/2) + output_title + '#'*int((terminal_width-len(output_title))/2))
            print('#' * 2*terminal_width)

            print(TRAJ)

            # Combine X_0 and PI
            ego_agent = agents[0]
            num_agents = TRAJ.shape[0]
            K = ego_agent._K

            ### Plots #####################
            fig, axes = plt.subplots(1, figsize=(12,12))
            t = np.linspace(0, 1, 200)
            colours = ['r', 'b', 'g', 'orange', 'm']
            lightColours = ['pink', 'lightblue', 'lightgreen', 'yellow']
            state = ['X', 'Y', 'Z']

            # Save the animation
            ### X-Y View ###################
            for i in range(num_agents):
                axes.plot(TRAJ[i, :, 0], TRAJ[i, :, 1], 'X-', markersize=8, color=f'{colours[i]}')                   # Trajectory
                axes.plot(TRAJ[i, 0, 0], TRAJ[i, 0, 1], f'o-', color=f'{colours[i]}', markersize=8, label=f"UAV{i}") # Current Position
                axes.plot(agents[i].P_DES[0], agents[i].P_DES[1], f'D', markersize=10, color=f'{colours[i]}')        # Goal

                if i == 0 and ego_agent.kc > -1:
                    # axes.plot(TRAJ[i, ego_agent.kc+1, 0], TRAJ[i, ego_agent.kc+1, 1], 'o', color=f'{colours[i]}')                                                                   # Position at collision time
                    neighbourhood_i = patches.Circle((TRAJ[i, ego_agent.kc+1, 0], TRAJ[i, ego_agent.kc+1, 1]), ego_agent.neighbourhood,
                                                    edgecolor=f'{colours[i]}', facecolor='none', linewidth=2)                                           # neighbourhood
                    axes.add_patch(neighbourhood_i)
                elif i != 0:
                    # axes.plot(TRAJ[i, 0, 0], TRAJ[i, 0, 1], 'o', color=f'{colours[i]}')                                                                   # Position at collision time
                    neighbourhood_i = patches.Circle((TRAJ[i, 0, 0], TRAJ[i, 0, 1]), ego_agent.neighbourhood,
                                                      edgecolor=f'{colours[i]}', facecolor='none', linewidth=2)                                           # neighbourhood
                    axes.add_patch(neighbourhood_i)

            axes.axis("equal")
            axes.grid(True)
            axes.set_xlabel('X [m]')
            axes.set_ylabel('Y [m]')
            axes.set_title(f"Output")
            axes.legend()
            plt.tight_layout()
            plt.show()
            
            print()
            print('#' * 2*terminal_width)
            print()
            return 0

#################################################################################
def play_horizon(history, agents):
    ### Plots #####################
    fig, axes = plt.subplots(1, figsize=(12,12))
    colours = ['r', 'b', 'g', 'orange', 'm']
    lightColours = ['pink', 'lightblue', 'lightgreen', 'yellow']

    metadata = {'title': 'MPC Animation', 'artist': 'Nathan', 'comment': 'See the associated params.urdf file for details'}
    writer = FFMpegWriter(fps=20, metadata=metadata)

    mp4_name = "temp.mp4"


    # Save the animation
    with writer.saving(fig, mp4_name, dpi=100):
        for i in range(0, int(history.shape[0]), 1):

            axes.plot(history[:i, 0], history[:i, 1], f'o-', color=f'{colours[0]}', markersize=8, label=f"UAV{i}") # Agent Trajectory
            axes.plot(agents[0].P_DES[0], agents[0].P_DES[1], f'D', markersize=10, color=f'{colours[0]}')          # Goal

            axes.plot(agents[1].X_0[0], agents[1].X_0[1], 'o', color=f'{colours[1]}')
            neighbourhood_i = patches.Circle((agents[1].X_0[0], agents[1].X_0[1]), agents[1].neighbourhood,
                                             edgecolor=f'{colours[1]}', facecolor='none', linewidth=2)                                           # neighbourhood
            axes.add_patch(neighbourhood_i)

            axes.axis("equal")
            axes.grid(True)
            axes.set_xlabel('X [m]')
            axes.set_ylabel('Y [m]')
            axes.set_title(f"Output")
            # axes.legend()
            plt.tight_layout()
            

            writer.grab_frame()
            plt.pause(0.1)


if __name__ == "__main__":
    main()
