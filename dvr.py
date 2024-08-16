import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

class DVR:
    def __init__(self,grid=[-0.5,0.5],v=lambda x:0.,m=1.,q=-1.,basis='sinc',hmat=[],vals=[],vecs=[],stab_lengths=[],stab_energies=[],sym=False,avoided_crossings=[],av_roots=[]):
        self.grid=np.array(grid)
        self.v=v
        self.m=m
        self.q=q
        self.basis=basis
        self.hmat=np.array(hmat)
        self.vals=np.array(vals)
        self.vecs=np.array(vecs)
        self.stab_lengths=np.array(stab_lengths)
        self.stab_energies=np.array(stab_energies)
        self.sym=sym
        self.avoided_crossings=np.array(avoided_crossings)
        self.av_roots=av_roots

    def get_kb(self):
        return 3.166811563e-6

    def make_hmat_fgh(self):
        N=len(self.grid)
        L=self.grid[-1]-self.grid[0]
        H=np.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                if i==j:
                    H[i,j]=((((np.pi*N)**2.0)/(6.0*self.m*(L**2.0)))*(1.0-(1.0/(N**2.0))))+(self.v(self.grid[i])*(i!=0 and i!=N-1))+((10.0**15)*(i==0 or i==N-1))
                else:
                    H[i,j]=((np.pi**2.0)/(self.m*(L**2.0)))*((-1)**(j-i))*(np.cos((np.pi*(j-i))/N)/(np.sin((np.pi*(j-i))/N)**2))
                    H[j,i]=H[i,j]
        self.hmat=H
        return H

    def make_hmat_sinc(self):
        N=len(self.grid)
        step=self.grid[1]-self.grid[0]
        H=np.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                if i==j:
                    H[i,j]=((np.pi**2)/(6*(step**2)*self.m))+self.v(self.grid[i])
                else:
                    H[i,j]=((-1)**(i-j))/(((i-j)**2)*(step**2)*self.m)
                    H[j,i]=H[i,j]
        self.hmat=H
        return H

    def solve_sys(self):
        if self.basis.strip().lower()=='sinc':
            self.make_hmat_sinc()
        elif self.basis.strip().lower()=='fgh':
            self.make_hmat_fgh()
        else:
            sys.quit('!!!{} is not a recognized dvr basis!!!'.format(basis))
        val,vec=np.linalg.eig(self.hmat)
        idx=val.argsort()[::1]
        val=val[idx]
        vec=vec[:,idx]
        self.vals=val
        self.vecs=vec
        return val,vec

    def plot_vecs(self,wv=[1]):
        for i in wv:
            plt.title('Eigenfunction {}'.format(i))
            plt.xlabel('position')
            plt.ylabel('wavefunction')
            plt.plot(self.grid,self.vecs[:,i-1])
            plt.show()

    def make_stab_data(self,len_add=10.,both_sides=False,num_roots=10,verbose=0):
        step=self.grid[1]-self.grid[0]
        in_start=self.grid[0]
        in_stop=self.grid[-1]
        num_points=int(len_add/step)+1
        lengths=[]
        energies=[]
        for i in range(num_roots):
            energies.append([])
        for i in range(num_points):
            if both_sides:
                grid=np.arange(in_start-(i*step),in_stop+(i*step)+(step/2.),step)
            else:
                grid=np.arange(in_start,in_stop+(i*step)+(step/2.),step)
            lengths.append(grid[-1]-grid[0])
            self.grid=grid
            vals,vecs=self.solve_sys()
            for j in range(num_roots):
                energies[j].append(vals[j])
            if verbose>=1:
                print('Calculation done at length {}'.format(lengths[-1]))
        self.stab_lengths=np.array(lengths)
        self.stab_energies=np.array(energies)
        return lengths,energies

    def make_stab_plot(self):
        sfs=1./np.power(self.stab_lengths,2)
        if self.sym:
            leng=len(self.stab_energies)
            if leng%2==0:
                eleng=int(leng/2)
                oleng=eleng
            else:
                eleng=int(leng/2)+1
                oleng=eleng-1
            for i in range(eleng):
                plt.plot(sfs,self.stab_energies[i*2])
            plt.title('Even')
            plt.show()
            for i in range(oleng):
                plt.plot(sfs,self.stab_energies[(i*2)+1])
            plt.title('Odd')
            plt.show()
        else:
            for i in self.stab_energies:
                plt.plot(sfs,i)
            plt.show()

    def detect_avoided_crossings(self,each_side=5,plot=False):
        sfs=1./np.power(self.stab_lengths,2)
        if self.sym:
            eleng=int((len(self.stab_energies)-1)/2)
            oleng=int((len(self.stab_energies)-2)/2)
            acs=[]
            for i in range(eleng):
                all_diffs=[]
                k=0
                for j in range(len(self.stab_energies[i*2])):
                    diff=self.stab_energies[(i+1)*2,j]-self.stab_energies[i*2,j]
                    all_diffs.append(diff)
                    if j==0:
                        continue
                    if j==1 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        continue
                    if k==1 and all_diffs[-2]>all_diffs[-1]:
                        k=0
                        continue
                    if k==0 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        acs.append(np.array([sfs[max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[i*2,max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[(i+1)*2,max(0,(j-1)-each_side):min(len(sfs),j+each_side)]]))
                        continue
            for i in range(oleng):
                all_diffs=[]
                k=0
                for j in range(len(self.stab_energies[(i*2)+1])):
                    diff=self.stab_energies[((i+1)*2)+1,j]-self.stab_energies[(i*2)+1,j]
                    all_diffs.append(diff)
                    if j==0:
                        continue
                    if j==1 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        continue
                    if k==1 and all_diffs[-2]>all_diffs[-1]:
                        k=0
                        continue
                    if k==0 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        acs.append(np.array([sfs[max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[(i*2)+1,max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[((i+1)*2)+1,max(0,(j-1)-each_side):min(len(sfs),j+each_side)]]))
                        continue
        else:
            acs=[]
            for i in range(len(self.stab_energies)-1):
                all_diffs=[]
                k=0
                for j in range(len(self.stab_energies[i])):
                    diff=self.stab_energies[i+1,j]-self.stab_energies[i,j]
                    all_diffs.append(diff)
                    if j==0:
                        continue
                    if j==1 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        continue
                    if k==1 and all_diffs[-2]>all_diffs[-1]:
                        k=0
                        continue
                    if k==0 and all_diffs[-1]>all_diffs[-2]:
                        k=1
                        acs.append(np.array([sfs[max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[i,max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[i+1,max(0,(j-1)-each_side):min(len(sfs),j+each_side)]]))
                        continue
        if plot:
            for i in acs:
                plt.scatter(i[0],i[1])
                plt.scatter(i[0],i[2])
            plt.show()
        self.avoided_crossings=acs
        return acs

    def gpa(self,porder=0,qorder=1,rorder=2,nr_tol=1.0e-5,sqrt_num_in_points=10,max_nr_iter=100,verbose=0.):
        def func(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def dfuncdx(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*params0[j]*(x**j))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*params0[j+porder]*(x**(j-1)))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(j*params0[j+porder+qorder+1]*(x**(j-1)))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def dfuncde(X,porder0,qorder0,params0):
            x,e=X
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            return (2*e*ppoly)+qpoly
        def ddfuncddx(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*j*params0[j]*(x**(j-1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*(j-1)*params0[j+porder]*(x**(j-2)))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(j*(j-1)*params0[j+porder+qorder+1]*(x**(j-2)))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def ddfuncdxde(X,porder0,qorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*params0[j]*(x**j))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*params0[j+porder]*(x**(j-1)))
            return (2*e*ppoly)+qpoly
        def e_plus(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return (-qpoly+np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        def e_minus(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return (-qpoly-np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        av_roots=[]
        for i in self.avoided_crossings:
            sfs=np.array(list(i[0])+list(i[0]))
            es=np.array(list(i[1])+list(i[2]))
            ydata=np.zeros(2*len(i[0]))
            params=np.array([1. for _ in range(porder+qorder+rorder+2)])
            popt,pcov=curve_fit(lambda X,*params:func(X,porder,qorder,rorder,params),(sfs,es),ydata,p0=params)
            step=(max(i[0][0],i[0][-1])-min(i[0][0],i[0][-1]))/(sqrt_num_in_points-1.)
            test_grid=np.arange(min(i[0][0],i[0][-1]),max(i[0][0],i[0][-1])+(step/2.),step)
            parity=[-1,1]
            roots=[]
            for j in test_grid:
                for k in test_grid:
                    for par in parity:
                        xguess=complex(j,par*k)
                        pguess=np.array([xguess,e_plus(xguess,porder,qorder,rorder,popt)])
                        mguess=np.array([xguess,e_minus(xguess,porder,qorder,rorder,popt)])
                        pconverged=False
                        mconverged=False
                        for l in range(max_nr_iter):
                            pfvec=np.array([func((pguess[0],pguess[1]),porder,qorder,rorder,popt),dfuncdx((pguess[0],pguess[1]),porder,qorder,rorder,popt)])
                            mfvec=np.array([func((mguess[0],mguess[1]),porder,qorder,rorder,popt),dfuncdx((mguess[0],mguess[1]),porder,qorder,rorder,popt)])
                            pkmat=np.array([[dfuncdx((pguess[0],pguess[1]),porder,qorder,rorder,popt),dfuncde((pguess[0],pguess[1]),porder,qorder,popt)],[ddfuncddx((pguess[0],pguess[1]),porder,qorder,rorder,popt),ddfuncdxde((pguess[0],pguess[1]),porder,qorder,popt)]])
                            mkmat=np.array([[dfuncdx((mguess[0],mguess[1]),porder,qorder,rorder,popt),dfuncde((mguess[0],mguess[1]),porder,qorder,popt)],[ddfuncddx((mguess[0],mguess[1]),porder,qorder,rorder,popt),ddfuncdxde((mguess[0],mguess[1]),porder,qorder,popt)]])
                            invpkmat=np.linalg.inv(pkmat)
                            invmkmat=np.linalg.inv(mkmat)
                            pdelta=-np.dot(invpkmat,pfvec)
                            mdelta=-np.dot(invmkmat,mfvec)
                            pguess+=pdelta
                            mguess+=mdelta
                            if np.sum(np.abs(pdelta))<nr_tol:
                                pconverged=True
                            if np.sum(np.abs(mdelta))<nr_tol:
                                mconverged=True
                            if pconverged and mconverged:
                                break
                        if pconverged:
                            not_same=True
                            for m in roots:
                                same_test=np.sum(np.abs(m-pguess))
                                if same_test<nr_tol:
                                    not_same=False
                                    break
                            if not_same:
                                roots.append(pguess)
                        if mconverged:
                            not_same=True
                            for m in roots:
                                same_test=np.sum(np.abs(m-mguess))
                                if same_test<nr_tol:
                                    not_same=False
                                    break
                            if not_same:
                                roots.append(mguess)
            av_roots.append(np.array(roots))
            if verbose>=2.:
                print('Avoided Crossing:')
                print(i)
                print('Roots:')
                print(av_roots[-1])
                print('')
        self.av_roots=av_roots
        return av_roots
 
if __name__ == "__main__":
    def v(x):
        #return 100.*np.exp(-10.*(x**2))
        #return 0.5*(x**2)
        #return 0.
        #return 100.*np.sin(100.*np.pi*(x-10.)/20.)
        return 10.*(np.exp(-((x+5.)**2))+np.exp(-((x-5.)**2)))
        #return (10.*np.exp(-((x+5.)**2)))+(1.*np.exp(-((x-5.)**2)))
    step=0.1
    start=-10.
    stop=10.
    m=1.
    q=-1.
    grid=np.arange(start,stop+(step/2.),step)
    print('')
    print('grid:')
    print(grid)
    print('')
    hosys=DVR(grid,v,m,q,'sinc',sym=True)
    hosys.solve_sys()
    print('Eigenvalues are:')
    print(hosys.vals)
    print('')
    #hosys.plot_vecs([1,2,3,4,5])
    hosys.make_stab_data(both_sides=True,verbose=1)
    hosys.make_stab_plot()
    hosys.detect_avoided_crossings(each_side=5,plot=False)
    hosys.gpa(3,4,5,verbose=3.)
    hosys.detect_avoided_crossings(each_side=5,plot=True)



