from numpy import *
from matplotlib.pyplot import * 

def solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, user_action=None):
	
	b  = float(b)
	
	x  = linspace(0,Lx,Nx+1)		# discretization arrays and step
	y  = linspace(0,Ly,Ny+1)		
	dx = x[1]-x[0]
	dy = y[1]-y[0]
	Nt = int(round(T/float(dt)))	
	t  = linspace(0,Nt*dt,Nt+1)

	xv = x[:,newaxis]	# for vector operations
	yv = y[newaxis,:]
	
	Cx2 = (dt/dx)**2	# help variables (short-hand notation)
	Cy2 = (dt/dy)**2			
	A   = 1/(1+b/2*dt)

	u   = zeros((Nx+1,Ny+1))	# solution arrays
	u_1 = zeros((Nx+1,Ny+1))
	u_2 = zeros((Nx+1,Ny+1))

	Ix = range(0, u.shape[0])	# index sets	
	Iy = range(0, u.shape[1])
	It = range(0, t.shape[0])

	for i in Ix:
		for j in Iy:
			u_1[i,j] = I(x[i],y[j])

	if user_action is not None:
		user_action(u_1, x, y, t, 0) #add arguments for vectorized version

	for n in It[1:-1]:			
		u = advance_scalar(u, u_1, u_2, V, f, q, b, x, y, t, Cx2, Cy2, A, dt, n) 		

		u_2, u_1, u = u_1, u, u_2
		
		if user_action is not None:
			user_action(u_1, x, y, t, n) #add arguments for vectorized version

def advance_scalar(u, u_1, u_2, V, f, q, b, x, y, t, Cx2, Cy2, A, dt, n): #must be modified to include check for step1
	Ix = range(0,u.shape[0]); Iy = range(0,u.shape[1])
	if n==1:
		D1=0
		D2=2*dt	#Adjusting help variables for step 1
		B=0.5 
	else:
		D1=1	#Standard values for help variables
		D2=0	
		B=A
	for i in Ix:
		for j in Iy:
			im1=i-1	#introducing variables for indexes to handle boundary conditions
			ip1=i+1
			jm1=j-1
			jp1=j+1	
			if i==Ix[0]: 
				im1=ip1	#re-indexing to account for boundary conditions at x=0, x=Lx
			if i==Ix[-1]:
				ip1=im1
			if j==Iy[0]: 
				jm1=jp1	
			if j==Iy[-1]:
				jp1=jm1			
			#incrementation scheme
			u_xx=Cx2*(0.5*(q(x[ip1],y[j])+q(x[i],y[j]))*(u_1[ip1,j]-u_1[i,j])-0.5*(q(x[i],y[j])+q(x[im1],y[j]))*(u_1[i,j]-u_1[im1,j])) 
			u_yy=Cy2*(0.5*(q(x[i],y[jp1])+q(x[i],y[j]))*(u_1[i,jp1]-u_1[i,j])-0.5*(q(x[i],y[j])+q(x[i],y[jm1]))*(u_1[i,j]-u_1[i,jm1]))
 			
			u[i,j]=B*(2*u_1[i,j]-(1-b*dt/2)*(D1*u_2[i,j]-D2*V(x[i],y[j]))+u_xx+u_yy+f(x[i],y[j],t[n])*dt**2)
	return u
	
	
	
import nose.tools as nt

def test_constant_solution():
	u_exact = lambda x,y,t : zeros((size(x),size(y),size(t)))+2
	I = lambda x,y: zeros((size(x),size(y)))+2
	V = lambda x,y: zeros((size(x),size(y)))
	q = lambda x,y: 3+x+y 
	f = lambda x,y,t: zeros((size(x),size(y),size(t)))
	
	b  = 2	
	Lx = 4
	Ly = 4
	Nx = 4
	Ny = 4
	dt = 1
	T  = 4
	
	def assert_no_error(u,x,y,t,n):
		u_e  = u_exact(x,y,t[n])
		diff = abs(u-u_e).max()
		nt.assert_almost_equal(diff,0,places=13)
	
	solver(I,V,f,q,b,Lx,Ly,Nx,Ny,dt,T,user_action=assert_no_error)


	








    	 
