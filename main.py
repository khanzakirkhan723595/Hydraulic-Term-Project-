from re import X
import autograd as ad 

from autograd import grad, jacobian

import autograd.numpy as np

#Equations of at nodes
f1= lambda x: x[0] + 0*x[1] + 0*x[2] + 0*x[3] + 0*x[4] + x[5] + 0*x[6] - 2.5
f2= lambda x: -1*x[0] + x[1] + 0*x[2] + 0*x[3] + 0*x[4] + 0*x[5] + x[6] + 0
f3= lambda x: 0*x[0] + -1*x[1] + x[2] + 0*x[3] + 0*x[4] + 0*x[5] + 0*x[6] + 0.5
f4= lambda x: 0*x[0] + 0*x[1] + -1*x[2] + -1*x[3] + 0*x[4] + 0*x[5] + 0*x[6] + 1
f5= lambda x: 0*x[0] + 0*x[1] + 0*x[2] + x[3] + -1*x[4] + 0*x[5] + -1*x[6] + 1
#Equation of the loops
f6= lambda x: 0*(x[0]**2) + 130703.32*(x[1]**2) + 43567.77*(x[2]**2) + -130703.32*(x[3]**2) + 0*(x[4]**2) + 0*(x[5]**2) - 330842.79*(x[6]**2) + 0
f7= lambda x: 10163.49*(x[0]**2) + 0*(x[1]**2) + 0*(x[2]**2) + 0*(x[3]**2) + 130703.32*(x[4]**2) - 10338.84*(x[5]**2) + 330842.79*(x[6]**2) + 0
#finding the jacobian of above functions
jac_f1=jacobian(f1)
jac_f2=jacobian(f2)
jac_f3=jacobian(f3)
jac_f4=jacobian(f4)
jac_f5=jacobian(f5)
jac_f6=jacobian(f6)
jac_f7=jacobian(f7)


i=0
error=1000
tol= 0.0001
#iteration
maxiter=1000

#7X7 Matrix 
M=7
N=7

#dtype=float).reshape(N,1) initializes a 7x1 Numpy array x_0 with initial values of 1 in each element.
#The dtype=float argument ensures that the values are of float type.
#method reshapes the 1D array into a 7x1 column vector
x_0 =np.array([1,1,1,1,1,1,1],dtype=float).reshape(N,1)

#performing Newton_Raphson for calculating the Q values in each pipe
while np.any(abs(error)>tol) and i<maxiter:
  fun_evaluate= np.array([f1(x_0),f2(x_0),f3(x_0),f4(x_0),f5(x_0),f6(x_0),f7(x_0)]).reshape(M,1)
  flat_x_0  = x_0.flatten()
  jac=np.array([jac_f1(flat_x_0),jac_f2(flat_x_0),jac_f3(flat_x_0),jac_f4(flat_x_0),jac_f5(flat_x_0),jac_f6(flat_x_0),jac_f7(flat_x_0)])
  jac=jac.reshape(N,M)

  x_new = x_0 - np.linalg.inv(jac)@fun_evaluate

  error= x_new - x_0

  x_0=x_new

  print(i)
  print(error)
  print("--------------------------")

  i=i+1

print("The Solution is")
print(x_new)
print("Q(AB)",x_new[0],"m3/s")
print("Q(BC)",x_new[1],"m3/s")
print("Q(CD)",x_new[2],"m3/s")
print("Q(DE)",x_new[3],"m3/s")
print("Q(EF)",x_new[4],"m3/s")
print("Q(FA)",x_new[5],"m3/s")
print("Q(BE)",x_new[6],"m3/s")

#making array for length, diameter & elvation 
l=np.array([600,600,200,600,600,200,200])
d=np.array([0.25,0.15,0.1,0.15,0.15,0.2,0.1])
elv=np.array([30,25,20,20,22,25])
f=0.2
head_init=15

#head calculation
print("head at node 1 is = [" ,head_init,"]m")
for i in range(5):
  hf_nxt = (16 * f * l[i]*(x_new[i])**2) / (2 * 9.81 * d[i]**5 * 3.14 ** 2)
  if elv[i]>elv[i+1]:
    head=head_init-hf_nxt

    #print("hf_next is",hf_nxt)
    print("head at node ",i+2,"is = " ,head,"m")
  else:
    head=head_init+hf_nxt
    print("head at node ",i+2,"is = " ,head,"m")
  head_init=head