import numpy as np
import matplotlib.pyplot as plt

def makeH(grid,v,m):
	N=len(grid)
	L=grid[-1]-grid[0]
	H=np.zeros((N,N))
	for i in range(N):
		for j in range(i+1):
			if i==j:
				H[i,j]=((((np.pi*N)**2.0)/(6.0*m*(L**2.0)))*(1.0-(1.0/(N**2.0))))+(v(grid[i])*(i!=0 and i!=N-1))+((10.0**15)*(i==0 or i==N-1))
			else:
				H[i,j]=((np.pi**2.0)/(m*(L**2.0)))*((-1)**(j-i))*(np.cos((np.pi*(j-i))/N)/(np.sin((np.pi*(j-i))/N)**2))
				H[j,i]=H[i,j]
	return H

def solveSys(grid,v,m):
	H=makeH(grid,v,m)
	val,vec=np.linalg.eig(H)
	idx=val.argsort()[::1]   
	val=val[idx]
	vec=vec[:,idx]
	return val,vec

if __name__ == "__main__":
	def v(x):
		return 0.5*(x**2)
	step=0.1
	start=-10.
	stop=10.
	mass=1.
	num_graph=5
	grid=np.arange(start,stop+(step/2.),step)
	print('')
	print('grid:')
	print(grid)
	print('')
	val,vec=solveSys(grid,v,mass)
	print('Eigenvalues are:')
	print(val)
	print('')
	for i in range(min(len(vec),num_graph)):
		plt.title('Eigenfunction {}'.format(i+1))
		plt.xlabel('position')
		plt.ylabel('wavefunction')
		plt.plot(grid,vec[:,i])
		plt.show()



