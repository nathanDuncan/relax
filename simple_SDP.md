# Problem 
The goal is to generate a collision-free trajectory that drives a single ego agent to a target postion directly opposite an obstacle. We model the obstacle as a second-order cone and aim to solve the problem using a semi-definate program relaxation.

## The Agents 
The base problem formulation is proposed for multi-agent transitions in "*Trajectory Generation for Multiagent Point-To-Point  Transitions via Distributed Model Predictive Control*"

The agents are modeled as unit masses in $\mathbb{R}^3$, with double integrator dynamics. This simplified model of a quadrotor with underlying position controller is used to achieve faster computations. Higher-order models can be accommodated. 
We use $\mathbf{p}_i[k]$, $\mathbf{v}_i[k]$, $\mathbf{a}_i[k]$ to represent the discretized $x$, $y$, $z$ position, velocity and accelerations of agent $i$ at time step $k$, where accelerations are the inputs. With a discretization step $h$, the dynamic equations are given by
$$
\mathbf{p}_i[k+1] = \mathbf{p}_i[k] + h\mathbf{v}_i[k] + \frac{h^2}{2}\mathbf{a}_i[k],
$$
$$
\mathbf{v}_i[k+1] = \mathbf{v}_i[k] + h\mathbf{a}_i[k]
$$

## Constraints
We may choose to constrain the motion of the agents to match the physics of the vehicle. First the agents have limited actuation, which bounds its minimum and maximum acceleration.
$$
\mathbf{a}_{min} <= \mathbf{a}_i[k] <= \mathbf{a}_{max}
$$
<!-- Secondly, the agents must remain within a volume (e.g. an indoor flying arena). We impose:
$$
\mathbf{p}_{min} <= \mathbf{p}_i[k] <= \mathbf{p}_{max}
$$ -->

## Collision Avoidance
The collision avoidance constraint is designed such that the agents safely traverse the environment. In the case of quadrotors, aerodynamic effects from neighbouring agents may lead to crashes. Thus, we model the collision boundary for each agent as an ellipsoid elongated along the vertical axis to capture the downwash effect of the agents' propellers, similar to [8]. The collision constraints between an agent $i$ and obstacle $j$ is defined,
$$
||\mathbf{\Theta}^{-1}(\mathbf{p}_i[k]-\mathbf{p}_j[k])||_n >= r_{min}
$$

where $n=2$ is the degree of the ellipsoid and $r_{min}$ is the minimum distance between agents in the $xy$ plane. The scaling matrix $\mathbf{\Theta}$ is defined as $\mathbf{\Theta}=$ diag$(a,b,c)$. We choose $a=b=1$ and $c>=1$. Thus the required minimum distance in the vertical axis is $r_{z, min} = cr_{min}$.

# Methodology
## The Agent Prediction Model
Using the dynmaics defined previously, we can develop a linear model to express the agents' states over a horizon of fixed length $K$. First we introduce the notation $\dot{(\cdot)}[k|k_t]$ which represents the predicted value of $(\cdot)[k_t + k]$ with the information available at $k_t$. In what follows, $k \in \{0, \dots, K-1\}$ is the descrete-time index of the prediction horizon. The dynamic model of agent $i$ is given in a compact representation as
$$
\hat{\mathbf{x}}_i[k+1|k_t] = \mathbf{A} \hat{\mathbf{x}}_i[k|k_t] + \mathbf{B}\hat{\mathbf{u}}_i[k|k_t]
$$
where $\hat{\mathbf{x}}_i \in \mathbb{R}^6$, $\mathbf{A} \in \mathbb{R}^{6 \times 6}$, $\mathbf{B} \in \mathbb{R}^{6 \times 3}$ and $\hat{\mathbf{u}}_i \in \mathbb{R}^3$.

Define the initial state at instant $k_t$, $\mathbf{X}_{0,i} = \mathbf{x}_i[k_t]$. Then we can write the position sequence $\mathbf{P}_i \in \mathbb{R}^{3K}$ as an affine function of the input sequence $\mathbf{U}_i \in \mathbb{R}^{3K}$,
$$
\mathbf{P}_i = \mathbf{A}_0 \mathbf{X}_{0, i} + \mathbf{\Lambda} \mathbf{U}_i
$$
where $\mathbf{\Lambda} \in \mathbb{R}^{3K \times 3K}$ is defined as
$$
\mathbf{\Lambda} = \begin{bmatrix}
                        \mathbf{\Psi B} & \mathbf{0}_3 & \dots & \mathbf{0}_3
                        \\
                        \mathbf{\Psi A B} &\mathbf{\Psi B} &  \dots & \mathbf{0}_3
                        \\\mathbf{P}_i = \mathbf{A}_0 \mathbf{X}_{0, i} + \mathbf{\Lambda} \mathbf{U}_i
                        \vdots & \ddots & \ddots & \vdots
                        \\
                        \mathbf{\Psi A}^{K-1} \mathbf{B} & \mathbf{\Psi A}^{K-2} \mathbf{B} &  \dots & \mathbf{\Psi B}
                   \end{bmatrix}
$$
with matrix $\mathbf{\Psi} = \begin{bmatrix} 
                                \mathbf{I}_3 & \mathbf{0}_3
                             \end{bmatrix}$
selecting the first three rows of the matrix products (those corresponding to the position states). Lastly, $\mathbf{A}_0 \in \mathbb{R}^{3K \times 6}$ reflects the propagation of the initial state, 
$$
\mathbf{A}_0 = \begin{bmatrix}
                (\mathbf{\Psi A})^T & (\mathbf{\Psi A}^2)^T & \dots & (\mathbf{\Psi A}^K)^T
               \end{bmatrix}^T
$$

$\mathbf{A}_0$ and $\mathbf{\Lambda}$ are created in the function, **get_model_prediction_matrices(K, A, B)**.

## Objective Function
The objective function that is minimized to compute the optimal input sequence has three main components: trajectory error, control effort, and input variation.

### Trajectory Error Penalty:
This term drives the agent to their goals. We aim to minimize the sum of errors between the positions at the last $\kappa$ time steps of the horizon and the desired final position $\mathbf{p}_{d,i}$. The error term is defined as
$$
e_i = \sum_{k=K-\kappa}^{K} || \hat{\mathbf{p}}_i[k|k_t] - \mathbf{p}_{d,i}||_2
$$
This term can be turned into a quadratic cost function in terms of the input sequence,
$$
J_{e,i}=\mathbf{U}_i^T(\mathbf{\Lambda}^T \tilde{\mathbf{Q}}\mathbf{\Lambda})\mathbf{U}_i - 2(\mathbf{P}_{d,i}^T \tilde{\mathbf{Q}}\mathbf{\Lambda}-(\mathbf{A}_0\mathbf{X}_{0,i})^T \tilde{\mathbf{Q}}\mathbf{\Lambda})\mathbf{U}_i
$$
where $\tilde{\mathbf{Q}} \in \mathbb{R}^{3K \times 3K}$ is a positive definite and block diagonal matrix that weights the error at each time step. A value of $\kappa = 1$ leads to $\tilde{\mathbf{Q}} = $ diag $(\mathbf{0}_3, \dots , \mathbf{Q})$ with matrix $\mathbf{Q} \in \mathbb{R}^{3 \times 3}$ chosen as a diagonal positive semidefinite matrix. Higher values of $\kappa$ lead to more aggressive agent behaviour with agents moving faster towards their goals, but may also lead to overshooting at the target location.

With this objective and the constraints discussed earlier, we obtain the following nonconvex quadratically constrained quadratic program,
$$
\begin{equation}
    \begin{matrix}
    minimize & \frac{1}{2}\mathbf{u}^T \mathbf{P} \mathbf{u} + \mathbf{q}^T \mathbf{u}
    \\
    \mathbf{u}
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{u} \le \mathbf{b}_{ineq}
    
    \end{matrix}
\end{equation}
$$

## On-demand Collision Avoidance
To implement on-demand collision avoidance, we leverage the predictive nature of DMP to detect colliding trajectories and impose constraints to avoid the *first* predicted collision. this strategy differs from [6] since we do not attempt to incrementally resolve *all* predicted collisions, only the most relevant one.
Agent $i$ detects a collision at time step $k_{c,i}$ of the previously considered horizon whenever the inequality
$$
\begin{equation}
    \xi_{ij} = ||\mathbf{\Theta}^{-1}(\hat{\mathbf{p}}_i[k_{c,i}|k_t-1]-\hat{\mathbf{p}}_j[k_{c,i}|k_t - 1])||_n \ge r_{min}
\end{equation}
$$
does not hold with a neighbour $j$. Note that at solving time $k_t$, the agents only have information of the other agents computed at $k_t - 1$, meaning that the collision is predicted to hppen at time $k_{c,i} +k_t -1$. In what follows, $k_{c,i}$ represents the first time step of the horizon where agent $i$ predicts a collison with any neighbour. We include collision constraints with the subset of agents $\mathbf{\Omega}_i$ defined as
$$
    \mathbf{\Omega}_i = \{j \in \{1, \dots, N\} | \xi_{ij} < f(r_{min}), i \ne j\},
$$

where $f(r_{min})$ models the radius around the agent, which defines the neighbours to be considered as obstacles when solving the problem. 

To augment the problem with collision avoidance capabilities, we add an additional quadratic constraint,
$$
        ||\mathbf{\Theta}^{-1}(\mathbf{\hat{p}}_i[k_{c,i}-1|k_t]-\mathbf{\hat{p}}_j[k_{c,i}|k_t-1])||_2 \ge r_{min}
$$
This constraint restricts the position of agent $i$ at horizon step $k_{c,i}-1$ to lie beyond an ellipsoidal region about the position of agent $j$ predicted in the previous time step $k_t -1$.

To format this constraint in terms of the optimization variables, (cardinal accelerations), $\mathbf{u}$, we express the predicted position as a linear transform of the inputs using the stacked transformation system, $\mathbf{P}_i = \mathbf{A}_0 \mathbf{X}_{0, i} + \mathbf{\Lambda} \mathbf{u}_i$ and a block diagonal selection matrix $\mathbf{S}_i \in \mathbb{R}^?$ to isolate the postion at the time of collision. Additionally we may write $\mathbf{\hat{p}}_j[k_{c,i}|k_t-1]$ shorthand as $\mathbf{\hat{p}}_j$.
Writing Equation~\eqref{eq:LuisEllipsoid}, in terms of the optimization variable, we are able to expand and simplify,
$$
   ||\mathbf{\Theta}^{-1}(\mathbf{S}_i(\mathbf{A}_0 \mathbf{X}_{0, i} + \mathbf{\Lambda} \mathbf{u}_i)-\mathbf{\hat{p}}_j)||_2 \ge r_{min}
$$
$$
   ||\mathbf{\Theta}^{-1}((\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} + \mathbf{S}_i\mathbf{\Lambda} \mathbf{u}_i)-\mathbf{\hat{p}}_j)||_2 \ge r_{min}
$$
$$
   ||\mathbf{\Theta}^{-1}((\mathbf{S}_i\mathbf{\Lambda} \mathbf{u}_i)+ (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j))||_2 \ge r_{min}
$$
Lets call the two inner terms $\gamma = \mathbf{S}_i\mathbf{\Lambda} \mathbf{u}_i + \mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j$.
$$
   ||\mathbf{\Theta}^{-1}(\gamma)||_2 \ge r_{min}
$$

Expanding the Euclidean norm by $||\mathbf{x}||_2 = \sqrt{\mathbf{x}^T\mathbf{x}}$, our constraint becomes,

$$
    \sqrt{(\mathbf{\Theta}^{-1}(\gamma))^T(\mathbf{\Theta}^{-1}(\gamma))} \ge r_{min}
$$
Squaring both sides,
$$
    (\mathbf{\Theta}^{-1}(\gamma))^T(\mathbf{\Theta}^{-1}(\gamma)) \ge r_{min}^2
$$
Since $\mathbf{\Theta}$ is a diagonal matrix $(\mathbf{\Theta}^{-1})^T \mathbf{\Theta}^{-1} = \mathbf{\Theta}^{-2}$, simplifying the expression to:
$$
    \gamma^T\mathbf{\Theta}^{-2}\gamma \ge r_{min}^2
$$
Then we are able to reintroduce our optimization variable,
$$
    (\mathbf{S}_i\mathbf{\Lambda} \mathbf{u}_i + \mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)^T\mathbf{\Theta}^{-2}(\mathbf{S}_i\mathbf{\Lambda} \mathbf{u}_i + \mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j) \ge r_{min}^2
$$
and expand,
$$
    (\mathbf{S}_i \mathbf{\Lambda}\mathbf{u}_i)^T \mathbf{\Theta}^{-2} (\mathbf{S}_i \mathbf{\Lambda}\mathbf{u}_i) 
    +2 (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)^T \mathbf{\Theta}^{-2} (\mathbf{S}_i \mathbf{\Lambda}\mathbf{u}_i)
    + (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)^T \mathbf{\Theta}^{-2} (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)
    \ge r_{min}^2
$$

the constraint then may be written in affine form,
$$
    \begin{bmatrix} 
        \mathbf{\mathbf{u}_i} \\ 1
    \end{bmatrix}^T
    \begin{bmatrix} 
        \mathbf{A}_{eps} & \mathbf{b}_{eps} \\
        \mathbf{b}_{eps}^T & c_{eps}
    \end{bmatrix}
    \begin{bmatrix} 
        \mathbf{\mathbf{u}_i} \\ 1
    \end{bmatrix} \ge 0
$$
where $\mathbf{A}_{eps} = \mathbf{\Lambda}^T \mathbf{S}_i^T \mathbf{\Theta}^{-2} \mathbf{S}_i \mathbf{\Lambda}$,
 $\mathbf{b}_{eps} = (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)^T \mathbf{\Theta}^{-2} \mathbf{S}_i \mathbf{\Lambda}$
  and $c_{eps} = (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j)^T \mathbf{\Theta}^{-2} (\mathbf{S}_i\mathbf{A}_0 \mathbf{X}_{0, i} -\mathbf{\hat{p}}_j) - r_{min}^2$.

Augmenting Problem~\eqref{eq:QP} with the constraint developed in Equation~\eqref{eq:QC}, we obtain the following nonconvex quadratically constrained quadratic program,
$$
\begin{equation} 
    \begin{matrix}
    minimize & \frac{1}{2}\mathbf{u}_i^T \mathbf{P} \mathbf{u}_i + \mathbf{q}^T \mathbf{u}_i
    \\
    \mathbf{u}_i
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{u}_i \le \mathbf{b}_{ineq}
    \\
& \mathbf{u}_i^T\mathbf{A}_{eps}\mathbf{u}_i - 2\mathbf{b}_{eps}\mathbf{u}_i + c_{eps} \ge 0
    \end{matrix}
\end{equation}
$$

## Semidefinite Programming Relaxation
To manage the nonconvex ellipsoid constraint, we will attempt to use Shor's semidefinite relaxation \cite{shor1987quadratic}, essentially a first-order relaxation particularly designed for quadratically constrained quadratic programs.
Shor's idea is to raise the problem into a higher-dimensional space by introducing a new matrix variable:
$$
    \mathbf{\mathcal{U}} = \mathbf{u}\mathbf{u}^T 
$$

With this new variable we may express Problem~\eqref{eq:QCQP}'s quadratic terms $\mathbf{u}^T\mathbf{A}_i\mathbf{u}$ as $\mathbf{tr}(\mathbf{A}_i\mathbf{\mathcal{U}})$,


<!-- \label{eq:SDP.1} -->
$$
\begin{equation}
    \begin{matrix}
    minimize & \frac{1}{2}\mathbf{tr}(\mathbf{P}\mathbf{\mathcal{U}}) + \mathbf{q}^T \mathbf{\mathbf{u}}
    \\
    \mathbf{u}, \mathbf{\mathcal{U}}
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{\mathbf{u}} \le \mathbf{b}_{ineq}
    \\
    & \mathbf{tr}(\mathbf{A}_{eps},\mathcal{U}) + 2\mathbf{b}_{eps}^T\mathbf{u}+c_{eps} \ge 0
    \\
    & \mathcal{U} = \mathbf{u}\mathbf{u}^T 
    \end{matrix}
\end{equation}
$$

Now the problem has a linear objective function, a linear equality constraint, a linear inequality constraint for system limitations, another linear inequality constraint for collision avoidance, and a nonlinear equality constraint $\mathcal{U} = \mathbf{u}\mathbf{u}^T$. 

However the rank constraint is non-convex, so we relax it by simply requiring $\mathcal{U}$ to be positive semidefinite, replacing the nonlinear equality constraint by an in equality $\mathcal{U} \succeq \mathbf{u}\mathbf{u}^T$ \cite{boyd2004convex}:

$$
\begin{matrix}
    minimize & \frac{1}{2}\mathbf{tr}(\mathbf{P}\mathbf{\mathcal{U}}) + \mathbf{q}^T \mathbf{\mathbf{u}}
    \\
    \mathbf{u}, \mathbf{\mathcal{U}}
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{\mathbf{u}} \le \mathbf{b}_{ineq}
    \\
    & \mathbf{tr}(\mathbf{A}_{eps},\mathcal{U}) + 2\mathbf{b}_{eps}^T\mathbf{u}+c_{eps} \ge 0
    \\
    & \mathcal{U} \succeq \mathbf{u}\mathbf{u}^T 
\end{matrix}
$$
This is a relaxation of ~\eqref{eq:SDP.1} since we have replaced one of the constraints with a looser constraint. As a final step we will express the new inequality as a linear matrix inequality by using Schur complement giving,
<!-- \label{eq:SDP} -->
$$
\begin{equation} 
    \begin{matrix}
    minimize & \frac{1}{2}\mathbf{tr}(\mathbf{P}\mathbf{\mathcal{U}}) + \mathbf{q}^T \mathbf{\mathbf{u}}
    \\
    \mathbf{u}, \mathbf{\mathcal{U}}
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{\mathbf{u}} \le \mathbf{b}_{ineq}
    \\
    & \mathbf{tr}(\mathbf{A}_{eps},\mathcal{U}) + 2\mathbf{b}_{eps}^T\mathbf{u}+c_{eps} \ge 0
    \\
    & \begin{bmatrix}
        \mathcal{U} & \mathbf{u} \\
        \mathbf{u}^T & 1
      \end{bmatrix} \succeq 0
    \end{matrix}
\end{equation}
$$

We seek a solution to $\mathbf{\mathcal{U}}$ that is of rank 1 due to the proposition of exactness from \href{https://hankyang.seas.harvard.edu/Semidefinite/Shor.html#semidefinite-relaxation-of-qcqps}{Harvard ENG-SCI 257: Semidefinite Optimization and Relaxation} \cite{SDOR}.

    .
Let $\mathbf{X}_*$ be an optimal solution to the SDP (3.4), if $rank(\mathbf{X}_*)=1$, then $\mathbf{X}_*$ can be factorized as $\mathbf{X}_* = \mathbf{x}_*\mathbf{x}_*^T$ with $\mathbf{x}_*$ a globally optimal solution to the QCQP (3.1). If so, we say the relaxation (3.4) is exact, or tight.

    .

The relaxation we are solving should be guaranteed to contain all feasible solutions to the original nonconvex problem however the sovler might still return a higher-rank solution due to the relaxation. In fact the solver will likely not find rank-1 solutions unless the problem naturally leads to rank-1 solutions (e.g. QCQP with a unique global optimum), or we encourage it via warm-starting, rank-penalty, or rounding heuristics.

To encourage $\mathbf{\mathcal{U}} \approx \mathbf{u}\mathbf{u}^T$ while keeping the problem convex, we choose to introduce a regluarization term to the objective that penalizes high rank,
$$
    \lambda \cdot tr(\mathbf{\mathcal{U}})
$$
This promotes low-rank $\mathbf{\mathcal{U}}$, indirectly encouraging rank-1 solutions.
Thus the complete problem is 
$$
\begin{equation} 
    \begin{matrix}
    minimize & \frac{1}{2}\mathbf{tr}(\mathbf{P}\mathbf{\mathcal{U}}) + \mathbf{q}^T +\lambda \cdot tr(\mathbf{\mathcal{U}})\mathbf{\mathbf{u}}
    \\
    \mathbf{u}, \mathbf{\mathcal{U}}
    \\
    s.t. &
    \mathbf{A}_{ineq} \mathbf{\mathbf{u}} \le \mathbf{b}_{ineq}
    \\
    & \mathbf{tr}(\mathbf{A}_{eps},\mathcal{U}) + 2\mathbf{b}_{eps}^T\mathbf{u}+c_{eps} \ge 0
    \\
    & \begin{bmatrix}
        \mathcal{U} & \mathbf{u} \\
        \mathbf{u}^T & 1
      \end{bmatrix} \succeq 0
    \end{matrix}
\end{equation}
$$