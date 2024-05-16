# Hydraulic-Term-Project-
This code is solving a problem related to a hydraulic network
consisting of 7 nodes connected by pipes, where the objective is
to find the flow rate in each pipe and the pressure head at each
node. The equations of the problem are defined as functions
using lambda expressions. The system of equations is solved using
Newton’s Raphson method with the help of the autograd library,
which provides automatic differentiation to compute the Jacobian
matrix. The solution is obtained by iterating until the error
between successive iterations is less than a certain tolerance.
The code then computes the pressure head at each node using
the flow rates obtained from the previous step, assuming that the
pipes are fully filled with water and that the losses due to friction
are proportional to the square of the flow rate
