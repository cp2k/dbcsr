import sys
import numpy as np
from make_twist_proper import make_twist_proper
#from matplotlib.pyplot import figure
#import matplotlib.pyplot as plt
#import module
import time

#np.__config__.show()

# modify only this to change the angle
#print (sys.argv[1])
m0=float(sys.argv[1])


r=1
a =2.46
d0=3.34

avec,bvec,data=make_twist_proper(a,m0,r)

print (avec)
print (bvec)
#print (data)
#avec,bvec,data=make_twist(a,1,1)
bvec=bvec*2.*np.pi



kvert1=[2./3.,1./3.]
kvert2=[  0.5,  0.0]
kvert3=[  0.0,  0.0]
kvert4=[1./3.,2./3.]
kvert5=[2./3.,1./3.]
kvert=np.array([kvert1,kvert2,kvert3,kvert4,kvert5])

# build kpath in Cartesian and lattice coordinates
kpath=[]
kpath_c=[]
dkc=[]
count=0
t2=0.0
nk_per_vert=8
for iv in range(len(kvert)-1):
   for ik in range(nk_per_vert):
      dkvert=kvert[iv+1,:]-kvert[iv,:]
      kp=kvert[iv,:]+dkvert[:]*float(ik)/float(nk_per_vert)
      kpc=np.matmul(kp,bvec)
      kpath.append(kp)
      kpath_c.append(kpc.tolist())
      count=count+1
      if (count==1):
        dkc.append(t2)
      else:
        k1=kpath_c[-1]
        k2=kpath_c[-2]
        k12=np.array(k1)-np.array(k2)
        t1=np.sqrt(k12[0]**2+k12[1]**2)
        t2=t2+t1
        dkc.append(t2)
      
nr=len(data)
nk=len(kpath)
kpathc=np.zeros((2,nk))
for ik in range(nk):
  kpathc[:,ik]=np.array(kpath_c[ik])

# prepare suspercell for R->k Fourier transform
ntrans=2
nrft=(2*ntrans+1)*(2*ntrans+1)
rft=np.zeros((2,nrft))
count=-1
for i1 in range(-ntrans,ntrans+1):
   for i2 in range(-ntrans,ntrans+1):
      count=count+1
      rft[:,count]=i1*avec[0,:]+i2*avec[1,:]
     
# for a r-vectors to atims in 3D
rr=np.zeros((3,nr))
ii=-1
for line in data:
   ii=ii+1
   temp=np.matmul(np.array([line[0],line[1]]),avec)
   x1=temp[0]
   y1=temp[1]
   z1=0.
   if (line[3]==2):
     z1=d0
   rr[0,ii]=x1
   rr[1,ii]=y1
   rr[2,ii]=z1


np.savetxt("kpath.dat",kpath)
np.savetxt("kpath_cart.dat",np.transpose(kpathc))
np.savetxt("recip_cell_vec.dat",bvec)

atoms_file = open("super_cell_vec.dat","w")
atoms_file.write("2\n")
np.savetxt(atoms_file, avec)
atoms_file.close()

atoms_file = open("coords.dat","w")
atoms_file.write(str(nr)+"\n")
np.savetxt(atoms_file, np.transpose(rr))
atoms_file.close()

np.savetxt("rft.dat", np.transpose(rft))
#atoms_file.close()

#print (len(rr[1]))
#print (rr)
#print (rft)


sys.exit("here")

