import sys
import numpy as np
import math
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure


def make_twist_proper(a,m,r):
  print ("m: ",m)
  print ("r: ",r)
  modr3=r%3

  costheta=(3*m**2+3*m*r+0.5*r**2)/(3*m**2+3*m*r+r**2)

  print ('costheta: ', costheta)
  theta=np.arccos(costheta)
  print ('theta (radians): ',theta)
  thetadeg=np.rad2deg(theta)
  print ('theta (degrees): ',thetadeg)
  # rotational matrices
  pi2=0.5*np.pi
  rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
  rot90=np.array([[np.cos(pi2),-np.sin(pi2)],[np.sin(pi2),np.cos(pi2)]])
  # direct lattice
  a1=np.array([a/2,a*math.sqrt(3)/2])
  a2=np.array([a,0])
  aa =np.vstack((a1,a2))
  aar=np.matmul(aa,rot)
  # reciprocal lattice
  b1=2.*math.pi/a  * np.array([1,-1/math.sqrt(3)])
  b2=2.*math.pi/a  * np.array([0,2./math.sqrt(3)])
  bb=np.vstack((b1,b2))
  bbr=np.matmul(bb,rot)
  # Superlattice vectors
  A1=[0,0]
  A2=[0,0]
  number_of_atoms=0
  if (modr3==0):
     n=r/3
     A1[0]=(m+n)*a1[0]+n*a2[0]
     A2[0]=-n*a1[0]+(m+2*n)*a2[0]
     A1[1]=(m+n)*a1[1]+n*a2[1]
     A2[1]=-n*a1[1]+(m+2*n)*a2[1]
     number_of_atoms=4*(m**2+m*r+r**2/3)
  else:
     A1[0]=m*a1[0]+(m+r)*a2[0]
     A2[0]=-(m+r)*a1[0]+(2*m+r)*a2[0]
     A1[1]=m*a1[1]+(m+r)*a2[1]
     A2[1]=-(m+r)*a1[1]+(2*m+r)*a2[1]
     number_of_atoms=4*(3*m**2+3*m*r+r**2)
 
  print ('Number of atoms: ',number_of_atoms)
  AA=np.vstack((np.array(A1),np.array(A2)))
  # Reciprocal of Superlattice vectors
  AA90=np.matmul(AA,rot90)
  BB1=AA90[0,:]
  BB2=AA90[1,:]
  B1=BB2/np.matmul(A1,BB2)
  B2=BB1/np.matmul(A2,BB1)
  BB=np.vstack((B1,B2))
  # transpose of reciprocal Moire lattice vectors
  BBT=BB.transpose()
  # Bernal 
  #atoms0=[[0.,0.,'A',1],[2./3.,2./3.,'B',1],[-2./3.,-2./3.,'A',2],[0.,0.,'B','2']]
  # AA stacking
  atoms0=[[0.,0.,'A',1],[2./3.,2./3.,'B',1],[0.,0.,'A',2],[2./3.,2./3.,'B',2]]
  
  ntrans=140
  data=[]

  eps=0.0000001
  for ix in range(-ntrans,ntrans+1):
     for iy in range(-ntrans,ntrans+1):
        for atline in atoms0:
           atl=np.array([atline[0]+float(ix),atline[1]+float(iy)])
           #print(atl,atline[0],atline[1])
           atc=[]
           # Cartesian cordinates
           if (atline[3]==1):
             # not-rotated layer
             atc=np.matmul(atl,aa)
           else:
             # rotated layer
             atc=np.matmul(atl,aar)
           # lattice corrdinates with respect to Moire lattice vecors
           atl_super=np.matmul(atc,BBT)
           # main array
           if (np.all(atl_super>=-eps) and np.all(atl_super<1.-eps)):
             atl_list=atl_super.tolist()
             data.append([atl_list[0],atl_list[1],atline[2],atline[3]])


  print ("len data: ",len(data)     )
  num_A1=0
  num_A2=0
  num_B1=0
  num_B2=0
  A1C=[]
  B1C=[]
  A2C=[]
  B2C=[]
  for line in data:
     temp=np.matmul(np.array([line[0],line[1]]),AA)
     temp=temp.tolist()
     if (line[3]==1):
       if (line[2]=='A'):
         num_A1=num_A1+1
         A1C.append(temp)
       else:
         num_B1=num_B1+1
         temp=np.matmul(np.array([line[0],line[1]]),AA)
         B1C.append(temp)
     else:
       if (line[2]=='A'):
         num_A2=num_A2+1
         A2C.append(temp)
       else:
         num_B2=num_B2+1
         temp=np.matmul(np.array([line[0],line[1]]),AA)
         B2C.append(temp)

  A1C=np.array(A1C)
  B1C=np.array(B1C)
  A2C=np.array(A2C)
  B2C=np.array(B2C)
       
  print ("len A1,B1,A2,B2:")
  print (num_A1," ",num_B1," ",num_A2,' ',num_B2)     

  #figure(num=None, figsize=(10,8), dpi=100, facecolor='w', edgecolor='k')

  #plt.xlim( 0, 180)
  #plt.ylim(-80, 80)
  #plt.gca().set_aspect('equal', adjustable='box')
  #ms1=30
  #ms2=10
  #plt.scatter(A1C[:,0], A1C[:,1], c='r',s=ms1)
  #plt.scatter(B1C[:,0], B1C[:,1], c='y',s=ms1)
  #plt.scatter(A2C[:,0], A2C[:,1], c='b',s=ms2)
  #plt.scatter(B2C[:,0], B2C[:,1], c='g',s=ms2)
  ##plt.quiver([0,0],[0,0],aa[:,0],aa[:,1],color=['r','r'],angles='xy',scale_units='xy',scale=1)a
  ##x1=(AA[0,0]+AA[1,0])/3
  ##y1=(AA[0,1]+AA[1,1])/3
  ##plt.quiver(0,0,x1,y1,color='r',angles='xy',scale_units='xy',scale=1,width=0.01)
  ##x2=x1*2
  ##y2=y1*2
  ##plt.quiver(0,0,x2,y2,color='b',angles='xy',scale_units='xy',scale=1,width=0.005)
  #plt.savefig('atoms.png', bbox_inches='tight',dpi=600)
  #plt.gcf().clear()
  #plt.show()
  
  # unify all the coordinates of the atoms
  return AA,BB,data
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
