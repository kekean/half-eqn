# Implement 2D half equation turbulence model
from fenics import *
import numpy as np
from ufl import *
from dolfin import *
import sympy as sp 


# dt = dt_arr[kk]
dt = .01

nu = 1/10000.


m=0
# 0: half equation model
# 1: full k equation model 
if m==0:
    method = "half_eqn"
if m==1:
    method = "full_eqn"
    

keyword = method
ufile = File('Vid_Folder/'+keyword+'/Paraview/velocity'+keyword+".pvd")
#Use time 0-10
T_0 = 0.0
T_f = 10 
t = T_0

TOL = 1.0e-15
t_num = int((T_f-T_0)/dt)

#Generalized Offset Cylinders
circle_outx = 0.0
circle_outy = 0.0
circle_outr = 1.0
circle_inx = 0.5
circle_iny = 0.0
circle_inr = 0.1

#Create Mesh and Domains
# N_outer = 400
# N_inner = 100

f = Expression(("mint*(-4*x[1]*(1-pow(x[0],2)-pow(x[1],2)))",\
                    "mint*(4*x[0]*(1-pow(x[0],2)-pow(x[1],2)))"),degree = 4, mint= 0.0)

# d = Expression("mint*(-4*x[1]*(1-pow(x[0],2)-pow(x[1],2)))",degree = 4, mint= 1.0)
# d = Expression('-abs(pow(x[0],2) + pow(x[1],2),.5)-(r_out+r_in)/2)+(r_out-r_in)/2',r_in = circle_inr , r_out =circle_outr,  degree=2) #wall distance formula
d = Expression(("-abs(pow(pow(x[0],2) + pow(x[1],2),.5))-(r_out+r_in)/2+(r_out-r_in)/2"),r_in = circle_inr , r_out =circle_outr,  degree=2)

#Setup inner products -----------------

def a_1(u,v):
    return inner(nabla_grad(u),nabla_grad(v))
def a_sym(u,v):
    return inner(.5*(nabla_grad(u)+nabla_grad(u).T),nabla_grad(v))    
def a_t(u,v):
    return inner(.5*(nabla_grad(u)+nabla_grad(u).T),nabla_grad(v))      
def b(u,v,w):
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))  
def convect(u,v,w):
    return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
    return inner(p,div(v))


mesh = Mesh("mesh/offsetmesh.xml")
#Wall normal distance


V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh,MixedElement([V,Q]))    

(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

(uf, pf) = TrialFunctions(W) #For bdf2 part I think
(vf, qf) = TestFunctions(W)


w = Function(W)

#Smooth bridge (to allow force to increase slowly)
def smooth_bridge(t):
    if(t>1+1e-14):
        return 1.0
    elif(abs(1-t)>1e-14):
        return np.exp(-np.exp(-1./(1-t)**2)/t**2)
    else:
        return 1.0


#Boundary Conditions
def u0_boundary(x, on_boundary):
    return on_boundary

noslip = Constant((0.0, 0.0))


mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]))

originpoint = OriginPoint()

bcu = DirichletBC(W.sub(0),noslip,u0_boundary)
bcp = DirichletBC(W.sub(1), 0.0, originpoint, 'pointwise')
bcs = [bcu,bcp]


#Time and Velocity Setup 
tnPlus1_solutions = Function(W)                                                                    
(unP1,pnP1) = tnPlus1_solutions.split(True)                                                 

tn_solutions = Function(W)                                                                         
(un,pn) = tn_solutions.split(True)    

tnMin1_solutions = Function(W)
(unM1,pnM1)=tnMin1_solutions.split(True)


# Get u0 and u1 so BDF2 works
u_init = Expression(('0','0'),degree = 2) 
unM1.assign(interpolate(u_init,W.sub(0).collapse()))
un.assign(interpolate(u_init,W.sub(0).collapse()))

print('GOT HERE')
exit()


if m_s == 0:
    # LHS
    a =  (1.0/(1.0*dt))*1.0*inner(u,v)*dx + nu*a_1(u,v)*dx - c(p,v)*dx + c(q,u)*dx + b(un ,u,v)*dx 
    #RHS
    F = (1.0/(1.0*dt))*1.0*inner(un,v)*dx +inner(f,v)*dx 

if m_s == 1:
    # LHS
    a =  (1.0/(1.0*dt))*1.0*inner(u,v)*dx + nu*a_1(u,v)*dx - c(p,v)*dx + c(q,u)*dx + b(2.0*un - unM1,u,v)*dx 
    #RHS
    F = (1.0/(1.0*dt))*1.0*inner(un,v)*dx  +inner(f,v)*dx 


if m_s ==2:
    #Set up problem for NSE (From Mike and Michael)
    # LHS
    a =  (1.0/(2.0*dt))*3.0*inner(u,v)*dx + nu*a_1(u,v)*dx - c(p,v)*dx + c(q,u)*dx + b(2.0*un - unM1,u,v)*dx 
    #RHS
    F = (1.0/(2.0*dt))*4.0*inner(un,v)*dx  - (1.0/(2.0*dt))*inner(unM1,v)*dx+inner(f,v)*dx 


# Blank arrays to store quantites of interest
eps_arr = np.zeros((t_num+1))
ND_arr = np.zeros((t_num+1))
KE_arr = np.zeros((t_num+1))
KE2_arr = np.zeros((t_num+1))
FU_arr = np.zeros((t_num+1))


#Time Stepping
count = 0
persec = int(1./dt+1e-15)
while t <= T_f + TOL:
    print('Numerical time level: t = ',t)

    mint_val = min(1,t)
    f.mint = mint_val

    A = assemble(a)
    B = assemble(F)

    [bc.apply(A,B) for bc in bcs]
    solve(A,w.vector(),B)
    (unP1,pnPlus1) = w.split(True)



    eps_arr[count]=nu*assemble(a_1(unP1,unP1)*dx)
    if m_s == 0:
        ND_arr[count]= assemble(inner(unP1-un,unP1-un)*dx)*(1.0/dt)
    if m_s == 1:
        ND_arr[count]= assemble(inner(unP1-un,unP1-un)*dx)*(1.0/dt)
    if m_s == 2:
        ND_arr[count]= assemble(inner(unP1-2*un+unM1,unP1-2*un+unM1)*dx)*(1.0/dt)

    KE_arr[count]=assemble(inner(unP1,unP1)*dx)
    KE2_arr[count]=assemble(inner(2*un-unM1,2*un-unM1)*dx)
    FU_arr[count]=assemble(inner(unP1,f)*dx)


    unM1.assign(un)
    pnM1.assign(pn)

    un.assign(unP1)
    pn.assign(pnPlus1)

    ufile<<unP1
    
    t += dt
    if (count%persec == 0):
        np.savetxt('Arr_Folder/'+keyword+'/eps'+keyword+'.txt',eps_arr)
        np.savetxt('Arr_Folder/'+keyword+'/ND'+keyword+'.txt',ND_arr)
        np.savetxt('Arr_Folder/'+keyword+'/KE'+keyword+'.txt',KE_arr)
        np.savetxt('Arr_Folder/'+keyword+'/KE2'+keyword+'.txt',KE2_arr)
        np.savetxt('Arr_Folder/'+keyword+'/FU.'+keyword+'txt',FU_arr)
    count += 1


np.savetxt('Arr_Folder/'+keyword+'/eps'+keyword+'.txt',eps_arr)
np.savetxt('Arr_Folder/'+keyword+'/ND'+keyword+'.txt',ND_arr)
np.savetxt('Arr_Folder/'+keyword+'/KE'+keyword+'.txt',KE_arr)
np.savetxt('Arr_Folder/'+keyword+'/KE2'+keyword+'.txt',KE2_arr)
np.savetxt('Arr_Folder/'+keyword+'/FU.'+keyword+'txt',FU_arr)




























