"""
Solve the equation 
	u_tt + bu_t = (q(x,y)u_x)_x + (q(x,y)u_y)_y + f(x,y,t)
for initial conditions
	u(x,y,0) = I(x,y)
	u_x(x,y,0) = V(x,y)
and homogenous von Neuman boundary conditions (du/dn=0) vy finite differences.
"""
from numpy import *
from matplotlib.pyplot import *
import time 

def solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, user_action=None, 
		   version='scalar'):

	t0 = time.clock()	

	x  = linspace(0, Lx, Nx+1) 	# spatial mesh
	y  = linspace(0, Ly, Ny+1)		
	xv = x[:,newaxis]			# for vector operations
	yv = y[newaxis,:]
  	dx = x[1] - x[0]			
	dy = y[1] - y[0]
	
	Nt = int(round(T/float(dt)))	#time discretization	
	t  = linspace(0, Nt*dt, Nt+1)	
	
	b = float(b);       A = 1/(1+b/2*dt)	# help variables 
	Cx2 = (dt/dx)**2;   Cy2 = (dt/dy)**2			

	u   = zeros((Nx+1, Ny+1))	# solution arrays
	u_1 = zeros((Nx+1, Ny+1))
	u_2 = zeros((Nx+1, Ny+1))

	Ix = range(0, u.shape[0])	# index sets	
	Iy = range(0, u.shape[1])
	It = range(0, t.shape[0])

	if version == 'scalar':
		for i in Ix:
			for j in Iy:
				u_1[i,j] = I(x[i], y[j])
	else:
		u_1[:,:] = I(xv,yv)
		V_a = V(xv, yv)	#Evalute at startup for vectorized version
		q_a = q(xv, yv)
		#These will be used for the incrementation step.
		q_a_px = 0.5*(concatenate((q_a[1:,:],q_a[-2,:][None,:]),axis=0) - q_a) 
		q_a_mx = 0.5*(q_a - concatenate((q_a[1,:][None,:],q_a[:-1,:]),axis=0))
		q_a_py = 0.5*(concatenate((q_a[:,1:],q_a[:,-2][:,None]),axis=1) - q_a)
		q_a_my = 0.5*(q_a - concatenate((q_a[:,1][:,None],q_a[:,:-1]),axis=1))		

	if user_action is not None:
		user_action(u_1, x, xv, y, yv, t, 0) 

	for n in It[0:-1]:
		if version == 'scalar':			
			u = advance_scalar(u, u_1, u_2, V, f, q, b, x, y, t, Cx2, Cy2, A, 
							   dt, n)
		else:
			f_a=f(xv, yv, t[n])
			u = advance_vector(u, u_1, u_2, V_a, f_a, q_a_px, q_a_mx, q_a_py, 
							   q_a_my, b, x, y, t, Cx2, Cy2, A, dt, n) 		
		u_2, u_1, u = u_1, u, u_2
		
		if user_action is not None:
			user_action(u_1, x, xv, y, yv, t, n)

	t1=time.clock()
	run_time=t1-t0
	print 'For version %s the run time was =%g' % (version,run_time)

def advance_scalar(u, u_1, u_2, V, f, q, b, x, y, t, Cx2, Cy2, A, dt, n):
	Ix = range(0, u.shape[0]); Iy = range(0, u.shape[1])
	if n==0:
		#Adjusting help variables for first step (n=0)
		D1 = 0;	D2 = 2*dt;	B = 0.5 
	else:
		#Standard values for help variables
		D1 = 1;	D2 = 0;		B = A  
	for i in Ix:
		for j in Iy:
			#used to handle boundary conditions
			im1 = i-1;	ip1 = i+1;	jm1 = j-1;	jp1 = j+1	
			if i==Ix[0]: 
				im1 = ip1	
			if i==Ix[-1]:
				ip1 = im1
			if j==Iy[0]: 
				jm1 = jp1	
			if j==Iy[-1]:
				jp1 = jm1			
			#incrementation scheme
			u_xx = Cx2*(0.5*(q(x[ip1],y[j])+q(x[i],y[j]))*(u_1[ip1,j]-u_1[i,j])\
				   - 0.5*(q(x[i],y[j])+q(x[im1],y[j]))*(u_1[i,j]-u_1[im1,j])) 
			u_yy = Cy2*(0.5*(q(x[i],y[jp1])+q(x[i],y[j]))*(u_1[i,jp1]-u_1[i,j])\
				   - 0.5*(q(x[i],y[j])+q(x[i],y[jm1]))*(u_1[i,j]-u_1[i,jm1]))
 			
			u[i,j] = B*(2*u_1[i,j]-(1-b*dt/2)*(D1*u_2[i,j]-D2*V(x[i],y[j]))\
					 + u_xx+u_yy+f(x[i],y[j],t[n])*dt**2)
	return u
	
def advance_vector(u, u_1, u_2, V_a, f_a, q_a_px, q_a_mx, q_a_py, q_a_my, b, x, 
				   y, t, Cx2, Cy2, A, dt, n):
	Ix = range(0, u.shape[0]);	Iy = range(0, u.shape[1])
	if n==0:
		#Adjusting first step (n=0)
		D1=0;	D2=2*dt;	B=0.5
	else:
		#Standard values for help variables
		D1=1;	D2=0;		B=A	
	#incrementation scheme
	u_1_px = concatenate((u_1[1:,:],u_1[-2,:][None,:]),axis=0)
	u_1_mx = concatenate((u_1[1,:][None,:],u_1[:-1,:]),axis=0)
	u_1_py = concatenate((u_1[:,1:],u_1[:,-2][:,None]),axis=1)
	u_1_my = concatenate((u_1[:,1][:,None],u_1[:,:-1]),axis=1)
	
	u_xx = Cx2*(q_a_px*(u_1_px-u_1) - q_a_mx*(u_1-u_1_mx))
	u_yy = Cy2*(q_a_py*(u_1_py-u_1) - q_a_my*(u_1-u_1_my))
	
	u=B*(2*u_1 - (1-b*dt/2)*(D1*u_2-D2*V_a) + u_xx + u_yy + f_a*dt**2)
	return u	
	
import nose.tools as nt

def test_constant_solution():
	u_exact = lambda x,y,t : zeros((size(x),size(y),size(t))) + 2
	I = lambda x,y: zeros((size(x),size(y))) + 2
	V = lambda x,y: zeros((size(x),size(y)))
	q = lambda x,y: 3 + x + y 
	f = lambda x,y,t: zeros((size(x),size(y),size(t)))
	
	b  = 2	
	Lx = 4
	Ly = 4
	Nx = 4
	Ny = 4
	dt = 1
	T  = 4
	
	def assert_no_error(u, x, xv, y, yv, t, n):
		u_e  = u_exact(xv, yv, t[n])
		diff = abs(u-u_e).max()
		nt.assert_almost_equal(diff, 0, places=13)

	solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, user_action=assert_no_error,
		   version='scalar')

	solver(I,V,f,q,b,Lx,Ly,Nx,Ny,dt,T,user_action=assert_no_error,
		   version='vector')

