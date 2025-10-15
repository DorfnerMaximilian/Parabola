#Import all Packages used by this module
import numpy as np
from .PhysConst import ConversionFactors
import os 
from ctypes import c_char_p, cdll, POINTER, c_double, c_int,Structure
from copy import deepcopy

#get the environmental variable 
pathtocp2k=os.environ["cp2kpath"]
pathtocpp_lib="/home/nisarg3/software/Parabola/parabola/CPP_Extension/bin/AtomicBasis.so"
#Python Implementation of the Overlap for Normalization of the Basis
def gamma(alpha,n):
    ## computes the analytical value of the integral int_{-\inf}^{inf}x^ne^{-alphax^2}
    ## (see Manuscript gamma function)
    ## input:       alpha       gaussian exponent                   (float)
    ##              n           power of x in the integral          (int)
    ## output:      value       the actual value of the integral    (float)
    def doublefactorial(n):
        if n == -1:
            return 1
        elif n==0:
            return 1
        else:
            return n * doublefactorial(n-2)
    value=0.0
    if n%2==0:
        value=(doublefactorial(n-1)*np.sqrt(np.pi))/(2**(0.5*n)*alpha**(0.5*n+0.5))
    return value
#-------------------------------------------------------------------------
def Kcomponent(Y1k,Y2k,ik,jk,alpha):
    ## computes the analytical value of the integral int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## for one component k
    ## (see Manuscript K-Function function)
    ## input:       alpha        gaussian exponent                  (float)
    ##              Y1k          displacement Y1k                   (float)
    ##              Y2k          displacement Y2k                   (float)
    ##              ik           power of x-Y1k in the integral     (int)
    ##              jk           power of x-Y2k in the integral     (int)
    ## output:      sum          the  value of the integral         (float)
    def binom(n,k):
        return np.math.factorial(n)/np.math.factorial(k)/np.math.factorial(n-k)
    sum=0.0
    if Y1k==0.0 or Y2k==0.0:
        sum=gamma(alpha,ik+jk)
    else:
        for o in range(ik+1):
            for p in range(jk+1):
                if ik==o and jk==p:
                    sum+=gamma(alpha,o+p)
                else:
                    sum+=gamma(alpha,o+p)*binom(ik,o)*binom(jk,p)*(-Y1k)**(ik-o)*(-Y2k)**(jk-p)
    return sum
#-------------------------------------------------------------------------
def KFunction(Y1,Y2,iis,jjs,alpha):
    # the full K function iis=(ix,iy,iz) jjs=(jx,jy,jz)
    ## Computes the analytical value of full K function:
    ##  K=\prod_{k=1}^3int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## (see Manuscript K-Function function)
    ## input:       alpha           gaussian exponent                  (float)
    ##              Y1              displacement Y1                    (np.array)
    ##              Y2k             displacement Y2                    (np.array)
    ##              iis=(ix,iy,iz)  monomial decomposition             (np.array)
    ##              jjs=(jx,jy,jz)  monomial decomposition             (np.array)
    ## output:      output             the  value of the integral      (float)
    return np.prod([Kcomponent(Y1[it],Y2[it],iis[it],jjs[it],alpha) for it in range(len(Y1))])
#-------------------------------------------------------------------------
def JInt(X,lm1,lm2,A1,A2):
    # computes the J integral using the monomial decomposition of the 
    # solid harmonics.
    #Input: X (numpy.array) of the difference vector R1-R2
    #A1: positive numerical
    #A2: positive numerical


    ###############################################################################################################################
    ###############################################################################################################################
    #Define the cs hash map for the coefficients of the solid harmonics to homigenious monomials
    #returns the representation of a given solid harmonics (l,m) in terms of homogenious monomials
    #input:    
    #format: 
    # cs
    # is=cs[it][0] is the representation of the monomial in terms of 
    # x^is[0]y^is[1]z^is[2]
    # and cs[it][1] the corresponding prefactor (consistent with CP2K convention)
    cs={}

    cs['s']=[[[0,0,0],0.5/np.sqrt(np.pi)]]

    cs['py']=[[[0,1,0],np.sqrt(3./(4.0*np.pi))]]
    cs['pz']=[[[0,0,1],np.sqrt(3./(4.0*np.pi))]]
    cs['px']=[[[1,0,0],np.sqrt(3./(4.0*np.pi))]]

    cs['d-2']=[[[1,1,0],0.5*np.sqrt(15./np.pi)]]
    cs['d-1']=[[[0,1,1],0.5*np.sqrt(15./np.pi)]]
    cs['d0']=[[[2,0,0],-0.25*np.sqrt(5./np.pi)],[[0,2,0],-0.25*np.sqrt(5./np.pi)],[[0,0,2],0.5*np.sqrt(5./np.pi)]]
    cs['d+1']=[[[1,0,1],0.5*np.sqrt(15./np.pi)]]
    cs['d+2']=[[[2,0,0],0.25*np.sqrt(15./np.pi)],[[0,2,0],-0.25*np.sqrt(15./np.pi)]]

    cs['f-3']=[[[2,1,0],0.75*np.sqrt(35./2./np.pi)],[[0,3,0],-0.25*np.sqrt(35./2./np.pi)]]
    cs['f-2']=[[[1,1,1],0.5*np.sqrt(105./np.pi)]]
    cs['f-1']=[[[0,1,2],np.sqrt(21./2./np.pi)],[[2,1,0],-0.25*np.sqrt(21./2./np.pi)],[[0,3,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f0']=[[[0,0,3],0.5*np.sqrt(7./np.pi)],[[2,0,1],-0.75*np.sqrt(7/np.pi)],[[0,2,1],-0.75*np.sqrt(7/np.pi)]]
    cs['f+1']=[[[1,0,2],np.sqrt(21./2./np.pi)],[[1,2,0],-0.25*np.sqrt(21./2./np.pi)],[[3,0,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f+2']=[[[2,0,1],0.25*np.sqrt(105./np.pi)],[[0,2,1],-0.25*np.sqrt(105./np.pi)]]
    cs['f+3']=[[[3,0,0],0.25*np.sqrt(35./2./np.pi)],[[1,2,0],-0.75*np.sqrt(35./2./np.pi)]]

    cs['g-4']=[[[3,1,0],0.75*np.sqrt(35./np.pi)],[[1,3,0],-0.75*np.sqrt(35./np.pi)]] 
    cs['g-3']=[[[2,1,1],9.0*np.sqrt(35./(2*np.pi))/4.0],[[0,3,1],-0.75*np.sqrt(35./(2.*np.pi))]] 
    cs['g-2']=[[[1,1,2],18.0*np.sqrt(5./(np.pi))/4.0],[[3,1,0],-3.*np.sqrt(5./(np.pi))/4.0],[[1,3,0],-3.*np.sqrt(5./(np.pi))/4.0]] 
    cs['g-1']=[[[0,1,3],3.0*np.sqrt(5./(2*np.pi))],[[2,1,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[0,3,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]] 
    cs['g0']=[[[0,0,4],3.0*np.sqrt(1./(np.pi))/2.0],[[4,0,0],9.0*np.sqrt(1./(np.pi))/16.0],[[0,4,0],9.0*np.sqrt(1./(np.pi))/16.0],[[2,0,2],-9.0*np.sqrt(1./np.pi)/2.0],[[0,2,2],-9.0*np.sqrt(1./np.pi)/2.0],[[2,2,0],9.0*np.sqrt(1./np.pi)/8.0]]
    cs['g+1']=[[[1,0,3],3.0*np.sqrt(5./(2*np.pi))],[[1,2,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[3,0,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]]
    cs['g+2']=[[[2,0,2],18.0*np.sqrt(5./(np.pi))/8.0],[[0,2,2],-18.*np.sqrt(5./(np.pi))/8.0],[[0,4,0],3.*np.sqrt(5./(np.pi))/8.0],[[4,0,0],-3.*np.sqrt(5./(np.pi))/8.0]]
    cs['g+3']=[[[1,2,1],-9.0*np.sqrt(35./(2*np.pi))/4.0],[[3,0,1],0.75*np.sqrt(35./(2.*np.pi))]]
    cs['g+4']=[[[4,0,0],3.0*np.sqrt(35./np.pi)/16.0],[[2,2,0],-18.0*np.sqrt(35./np.pi)/16.0],[[0,4,0],3.0*np.sqrt(35./np.pi)/16.0]]
    ###############################################################################################################################
    ###############################################################################################################################
    Y1=A2*X/(A1+A2)
    Y2=-A1*X/(A1+A2)
    Z1=cs[lm1]
    Z2=cs[lm2]
    integral=np.sum([Z1[it1][1]*Z2[it2][1]*KFunction(Y1,Y2,Z1[it1][0],Z2[it2][0],A1+A2) for it1 in range(len(Z1)) for it2 in range(len(Z2))])
    return integral
#-------------------------------------------------------------------------
def IInt(R1,A1,lm1,R2,A2,lm2):
    # computes the I integral using the J integral and the Gaussian prefactor
    # solid harmonics.
    #input: 
    #R1:    (numpy.array)       position of nucleii 1
    #R2     (numpy.array)       position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #A1:    (positive real)     exponent gaussian of function 1
    #A2:    (positive real)     exponent of gaussian of function 2
    X=R1-R2
    Jintegral=JInt(X,lm1,lm2,A1,A2)
    A12red=-A1*A2/(A1+A2)
    Exponent=A12red*np.dot(X,X)
    gaussianPrefactor=np.exp(Exponent)
    integral=gaussianPrefactor*Jintegral
    return integral
#------------------------------------------------------------------------- 
def getoverlap(R1,lm1,dalpha1,R2,lm2,dalpha2):
    #Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
    #input: 
    #R1:    (numpy.array)                                   position of nucleii 1
    #R2     (numpy.array)                                   position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1) 
    #dalpha1: (list of list)    specifies the first Gaussian type of wave function 
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #dalpha2: (list of list)    specifies the second Gaussian type of wave function 
    overlap=0.0
    overlap=np.sum([dalpha1[it1][1]*dalpha2[it2][1]*IInt(R1,dalpha1[it1][0],lm1,R2,dalpha2[it2][0],lm2) for it1 in range(len(dalpha1)) for it2 in range(len(dalpha2))])
    return overlap
#-------------------------------------------------------------------------
#For the Calculation of the TransitionDipolemoments

def OInt(X,lm1,lm2,A1,A2,k):
        # computes the J integral using the monomial decomposition of the 
        # solid harmonics.
        #Input: X (numpy.array) of the difference vector R1-R2
        #A1: positive numerical
        #A2: positive numerical
        cs={}

        cs['s']=[[[0,0,0],0.5/np.sqrt(np.pi)]]

        cs['py']=[[[0,1,0],np.sqrt(3./(4.0*np.pi))]]
        cs['pz']=[[[0,0,1],np.sqrt(3./(4.0*np.pi))]]
        cs['px']=[[[1,0,0],np.sqrt(3./(4.0*np.pi))]]

        cs['d-2']=[[[1,1,0],0.5*np.sqrt(15./np.pi)]]
        cs['d-1']=[[[0,1,1],0.5*np.sqrt(15./np.pi)]]
        cs['d0']=[[[2,0,0],-0.25*np.sqrt(5./np.pi)],[[0,2,0],-0.25*np.sqrt(5./np.pi)],[[0,0,2],0.5*np.sqrt(5./np.pi)]]
        cs['d+1']=[[[1,0,1],0.5*np.sqrt(15./np.pi)]]
        cs['d+2']=[[[2,0,0],0.25*np.sqrt(15./np.pi)],[[0,2,0],-0.25*np.sqrt(15./np.pi)]]

        cs['f-3']=[[[2,1,0],0.75*np.sqrt(35./2./np.pi)],[[0,3,0],-0.25*np.sqrt(35./2./np.pi)]]
        cs['f-2']=[[[1,1,1],0.5*np.sqrt(105./np.pi)]]
        cs['f-1']=[[[0,1,2],np.sqrt(21./2./np.pi)],[[2,1,0],-0.25*np.sqrt(21./2./np.pi)],[[0,3,0],-0.25*np.sqrt(21./2./np.pi)]]
        cs['f0']=[[[0,0,3],0.5*np.sqrt(7./np.pi)],[[2,0,1],-0.75*np.sqrt(7/np.pi)],[[0,2,1],-0.75*np.sqrt(7/np.pi)]]
        cs['f+1']=[[[1,0,2],np.sqrt(21./2./np.pi)],[[1,2,0],-0.25*np.sqrt(21./2./np.pi)],[[3,0,0],-0.25*np.sqrt(21./2./np.pi)]]
        cs['f+2']=[[[2,0,1],0.25*np.sqrt(105./np.pi)],[[0,2,1],-0.25*np.sqrt(105./np.pi)]]
        cs['f+3']=[[[3,0,0],0.25*np.sqrt(35./2./np.pi)],[[1,2,0],-0.75*np.sqrt(35./2./np.pi)]]

        cs['g-4']=[[[3,1,0],0.75*np.sqrt(35./np.pi)],[[1,3,0],-0.75*np.sqrt(35./np.pi)]] 
        cs['g-3']=[[[2,1,1],9.0*np.sqrt(35./(2*np.pi))/4.0],[[0,3,1],-0.75*np.sqrt(35./(2.*np.pi))]] 
        cs['g-2']=[[[1,1,2],18.0*np.sqrt(5./(np.pi))/4.0],[[3,1,0],-3.*np.sqrt(5./(np.pi))/4.0],[[1,3,0],-3.*np.sqrt(5./(np.pi))/4.0]] 
        cs['g-1']=[[[0,1,3],3.0*np.sqrt(5./(2*np.pi))],[[2,1,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[0,3,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]] 
        cs['g0']=[[[0,0,4],3.0*np.sqrt(1./(np.pi))/2.0],[[4,0,0],9.0*np.sqrt(1./(np.pi))/16.0],[[0,4,0],9.0*np.sqrt(1./(np.pi))/16.0],[[2,0,2],-9.0*np.sqrt(1./np.pi)/2.0],[[0,2,2],-9.0*np.sqrt(1./np.pi)/2.0],[[2,2,0],9.0*np.sqrt(1./np.pi)/8.0]]
        cs['g+1']=[[[1,0,3],3.0*np.sqrt(5./(2*np.pi))],[[1,2,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[3,0,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]]
        cs['g+2']=[[[2,0,2],18.0*np.sqrt(5./(np.pi))/8.0],[[0,2,2],-18.*np.sqrt(5./(np.pi))/8.0],[[0,4,0],3.*np.sqrt(5./(np.pi))/8.0],[[4,0,0],-3.*np.sqrt(5./(np.pi))/8.0]]
        cs['g+3']=[[[1,2,1],-9.0*np.sqrt(35./(2*np.pi))/4.0],[[3,0,1],0.75*np.sqrt(35./(2.*np.pi))]]
        cs['g+4']=[[[4,0,0],3.0*np.sqrt(35./np.pi)/16.0],[[2,2,0],-18.0*np.sqrt(35./np.pi)/16.0],[[0,4,0],3.0*np.sqrt(35./np.pi)/16.0]]

        Y1=A2*X/(A1+A2)
        Y2=-A1*X/(A1+A2)
        Z1=cs[lm1]
        Z2=cs[lm2]
        integral=0.0
        if k==0:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0]+1,is1[1],is1[2]),is2,A1+A2)
        elif k==1:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0],is1[1]+1,is1[2]),is2,A1+A2)
        elif k==2:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0],is1[1],is1[2]+1),is2,A1+A2)

        A12red=-A1*A2/(A1+A2)
        Exponent=A12red*np.dot(X,X)
        gaussianPrefactor=np.exp(Exponent)
        integral=gaussianPrefactor*integral
        return integral
#-------------------------------------------------------------------------
#------------------------------------------------------------------------- 
def getContribution(R1,lm1,dalpha1,R2,lm2,dalpha2,k):
    #Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
    #input: 
    #R1:    (numpy.array)                                   position of nucleii 1
    #R2     (numpy.array)                                   position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1) 
    #dalpha1: (list of list)    specifies the first Gaussian type of wave function 
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #dalpha2: (list of list)    specifies the second Gaussian type of wave function 
    overlap=0.0
    for obj1 in dalpha1:
        for obj2 in dalpha2:
            d1=obj1[1]
            alpha1=obj1[0]
            d2=obj2[1]
            alpha2=obj2[0]
            overlap+=d1*d2*OInt(R1-R2,lm1,lm2,alpha1,alpha2,k)
    return overlap

#-------------------------------------------------------------------------
def getAngularMomentumString(l,m):
    ## Transforms the angular momentum notation (l,m) into the 's', 'py','pz','px' ect. notation
    ## input:   l                       angular momentum quantum number            (int)
    ##          m                       magnetic quantum number                    (int)
    ## output:  s                       the s- notation for the (l,m) pair         (string)  
    if l==0:
        s='s'
    elif l==1 and m==-1:
        s='py'
    elif l==1 and m==0:
        s='pz'
    elif l==1 and m==1:
        s='px'
    elif l==2:
        if m>0:
            s='d'+'+'+str(m)
        else:
            s='d'+str(m)
    elif l==3:
        if m>0:
            s='f'+'+'+str(m)
        else:
            s='f'+str(m)
    elif l==4:
        if m>0:
            s='g'+'+'+str(m)
        else:
            s='g'+str(m)
    elif l==5:
        if m>0:
            s='h'+'+'+str(m)
        else:
            s='h'+str(m)
    else:
        print("Higher order not yet implemented")
    return s

#-------------------------------------------------------------------------
def getNormalizationfactor(alpha,l):
    ## Transformationfactor between the normalized contracted cartesian Basis set from the data directory,
    ## and the not normalized Basis set used in the QS routines of cp2k. 
    ## This means the contraction coefficients c_dd from the data directory are connected with
    ## those used in the QS routines c_QS via c_QS(alpha,l)=Output(alpha,l)*c_dd(alpha,l), 
    ## where Output is the output of this function.
    ## input:   alpha                       the exponent of the Gaussian            (float)
    ##          l                           the angular momentum quantum number     (int)
    ## output:  The transformation factor            
    return alpha**(0.5*l+0.75)*2**(l)*(2.0/np.pi)**(0.75)
#-------------------------------------------------------------------------

def getBasisSetName(path,cp2kpath=pathtocp2k):
    ## Reads in from the .inp file in path the Basis sets used. Parses the corresponding 
    ## data from the cp2kpath/data/Basis_Set file and returns this parsed data as a list.
    ## Each element in this list is a string of the corresponding line in the Basis set file
    ## input:   path                path to the folder of the .inp file         (string)
    ## (opt.)   cp2kpath            path to the cp2k folder                     (string)
    ## output:  BasisInfoReadin                                                 (list of strings)       

    #open the .inp file
    inpfile= [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inpfile) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    atoms=[]
    BasisSetNames=[]
    BasisSetFileName='empty'
    BasisSetNameFlag=False
    with open(path+"/"+inpfile[0],'r') as g:
        lines=g.readlines()
        for line in lines:
            if len(line.split())>0:
                if line.split()[0]=="BASIS_SET_FILE_NAME":
                    BasisSetFileName=line.split()[1]
                if line.split()[0]=="&KIND":
                    atoms.append(line.split()[1])
                    BasisSetNameFlag=True
                if line.split()[0]=="END" and line.split()[1]=="&KIND":
                    BasisSetNameFlag=False
                if BasisSetNameFlag and line.split()[0]=="BASIS_SET":
                    BasisSetNames.append(line.split()[1])
    atomStrings=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
    print("Used Basis:",BasisSetFileName)
    BasisInfoReadin=[]
    ReadinFlag=False
    for it in range(len(atoms)):
        with open(cp2kpath+'/data/'+BasisSetFileName,'r') as g:
            for l in g:
                if len(l.split())>=1:
                    numericflag=l.split()[0][0].isnumeric()
                if len(l.split())==2 and not numericflag:
                    if l.split()[0]==atoms[it] and l.split()[1]==BasisSetNames[it]:
                        ReadinFlag=True
                    if (l.split()[0] in atomStrings and l.split()[0] not in atoms)or(l.split()[1] !=BasisSetNames[it]):
                        ReadinFlag=False
                elif len(l.split())>2 and not numericflag:
                    if (l.split()[0]==atoms[it] and l.split()[1]==BasisSetNames[it]) or (l.split()[0]==atoms[it] and l.split()[2]==BasisSetNames[it]):
                        ReadinFlag=True
                    elif bool(l.split()[0] in atomStrings and l.split()[0] not in atoms) or (bool(l.split()[1] !=BasisSetNames[it]) or bool(l.split()[2] !=BasisSetNames[it])) :
                        ReadinFlag=False
                if ReadinFlag:
                    BasisInfoReadin.append(l)
    for it in range(len(BasisInfoReadin)):
        item=BasisInfoReadin[it]
        item=item[:-1]
        BasisInfoReadin[it]=item
    return BasisInfoReadin
#-------------------------------------------------------------------------
def getBasis(filename):
    ##Constructs the (non-orthorgonal) Basis used in the CP2K calculation 
    ## input:
    ## (opt.)   filename            path to the calculation folder                   (string)
    ## output:  Basis               dic. of lists of sublists. The keys of the dic. are
    ##                              the atomic symbols.
    ##                              list contains sublist, where each Basisfunction of the 
    ##                              considered atom corresponds the one sublist.
    ##                              sublist[0] contains the set index as a string. 
    ##                              sublist[1] contains the shell index as a string
    ##                              sublist[2] contains the angular momentum label 
    ##                              as a string (e.g. shellindex py ect.)
    ##                              sublist[3:] are lists with two elements.
    ##                              The first corresponds the the exponent of the Gaussian
    ##                              The second one corresponds to the contraction coefficient
    BasisInfoReadin=getBasisSetName(filename)
    atoms=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
    BasisSet={}
    atom='NotDefined'
    Atombasis=[]
    newAtomSetflag=False
    newSetflag=False
    readinBasisfunctionsflag=False
    setcounter=1
    linecounter=0
    NumberofExponents=0
    numberofBasissets=0
    for line in BasisInfoReadin:
        splitedline=line.split()
        if len(splitedline)>0:
            firstcaracter=splitedline[0]
            if readinBasisfunctionsflag:
                if linecounter==NumberofExponents+1:
                    readinBasisfunctionsflag=False
                    newSetflag=True
                    setcounter+=1
                else:
                    exponent=float(splitedline[0])
                    coefficientiter=1
                    it=0
                    for it1 in range(len(ls)):
                        l=ls[it1]
                        for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                            coefficient=float(splitedline[coefficientiter])
                            for m in range(-l,l+1):
                                Basisfunctions[it].append([exponent,getNormalizationfactor(exponent,l)*coefficient]) #
                                it+=1
                            coefficientiter+=1
                    if linecounter==NumberofExponents:
                        Atombasis.append(Basisfunctions)
                    linecounter+=1
            if newSetflag and setcounter<=numberofBasissets:
                minprincipalQuantumnumber=int(splitedline[0])
                lmin=int(splitedline[1])
                lmax=int(splitedline[2])
                ls=np.array([lmin+it for it in range(0,lmax-lmin+1)])
                NumberofExponents=int(splitedline[3])
                NumofangularmomentumBasisfunctions=[]
                for split in splitedline[4:]:
                    NumofangularmomentumBasisfunctions.append(int(split))
                if (lmax-lmin+1)!=len(NumofangularmomentumBasisfunctions):
                    ValueError("Number of Basis functions does not fit!")
                NumofangularmomentumBasisfunctions=np.array(NumofangularmomentumBasisfunctions)
                readinBasisfunctionsflag=True
                newSetflag=False
                shell=0
                Basisfunctions=[]
                for it1 in range(len(ls)):
                    l=ls[it1]
                    for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                        shell+=1
                        for m in range(-l,l+1):
                            Basisfunctions.append([str(setcounter),str(shell),getAngularMomentumString(l,m)])
                linecounter=1
            if  newAtomSetflag:
                numberofBasissets=int(firstcaracter)
                setcounter=1
                newAtomSetflag=False
                newSetflag=True
            if setcounter==numberofBasissets and linecounter==NumberofExponents+1:
                Basis=[]
                for it1 in range(len(Atombasis)):
                    for it2 in range(len(Atombasis[it1])):
                        Basis.append(Atombasis[it1][it2])
                BasisSet[atom]=Basis
            if firstcaracter in atoms:
                Atombasis=[]
                atom=firstcaracter
                newAtomSetflag=True
    #Normalize the Basis
    for key in BasisSet.keys():
        for it in range(len(BasisSet[key])):
            lm1=BasisSet[key][it][2][:]
            lm2=lm1
            dalpha1=BasisSet[key][it][3:]
            dalpha2=dalpha1
            #C
            #Normalize the Basis
            R1=np.array([0.0,0.0,0.0])
            R2=np.array([0.0,0.0,0.0])
            normfactor=1.0/np.sqrt(getoverlap(R1,lm1,dalpha1,R2,lm2,dalpha2))
            for it2 in range(len(BasisSet[key][it])-3):
                BasisSet[key][it][it2+3][1]*=normfactor
    return BasisSet


#-------------------------------------------------------------------------
def getTransformationmatrix(Atoms1, Atoms2, Basis, cell_vectors=[0.0, 0.0, 0.0], cutoff_radius=50, pathtolib=pathtocpp_lib):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms1              atoms of the first index
    ##                           list of sublists. 
    ##                           Each of the sublists has five elements. 
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##       Atoms2              Atoms of the second index
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the 
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string. 
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label 
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##     cutoff_radius (opt.)  defines the distance upto which the calculation of overlaps will be taken into account (Unit: Angstroem)
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    

    # Initialize the python lists for Basis Set 1
    atoms_set1 = []
    positions_set1 = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms1)):
        Atom_type1 = Atoms1[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = np.array(Atoms1[itAtom1][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R1)):
                positions_set1.append(R1[it1])
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1

    # Initialize the python lists for Basis Set 2
    atoms_set2 = []
    positions_set2 = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms2)):
        Atom_type2 = Atoms2[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = np.array(Atoms2[itAtom2][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R2)):
                positions_set2.append(R2[it1])
            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_T_Matrix = lib.get_T_Matrix
    get_T_Matrix.restype = POINTER(c_double)
    get_T_Matrix.argtypes = [POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_double),
                             c_int,
                             c_double
                             ]

    freeArray = lib.free_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set1_ptr = (c_char_p * len(atoms_set1))(*[s.encode("utf-8") for s in atoms_set1])
    positions_set1_ptr = (c_double * len(positions_set1))(*positions_set1)
    alphas_set1_ptr = (c_double * len(alphas_set1))(*alphas_set1)
    alphas_lengths_set1_ptr = (c_int * len(alphas_lengths_set1))(*alphas_lengths_set1)
    contr_coef_set1_ptr = (c_double * len(contr_coef_set1))(*contr_coef_set1)
    contr_coef_lengths_set1_ptr = (c_int * len(contr_coef_lengths_set1))(*contr_coef_lengths_set1)
    lms_set1_ptr = (c_char_p * len(lms_set1))(*[s.encode("utf-8") for s in lms_set1])

    atoms_set2_ptr = (c_char_p * len(atoms_set2))(*[s.encode("utf-8") for s in atoms_set2])
    positions_set2_ptr = (c_double * len(positions_set2))(*positions_set2)
    alphas_set2_ptr = (c_double * len(alphas_set2))(*alphas_set2)
    alphas_lengths_set2_ptr = (c_int * len(alphas_lengths_set2))(*alphas_lengths_set2)
    contr_coef_set2_ptr = (c_double * len(contr_coef_set2))(*contr_coef_set2)
    contr_coef_lengths_set2_ptr = (c_int * len(contr_coef_lengths_set2))(*contr_coef_lengths_set2)
    lms_set2_ptr = (c_char_p * len(lms_set2))(*[s.encode("utf-8") for s in lms_set2])

    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)

    # Converting the cutoff_radius into atomic units
    cutoff_radius = cutoff_radius * ConversionFactors['A->a.u.']

    # Call the C++ function
    OLP_array_ptr = get_T_Matrix(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),cutoff_radius)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)

    OLP = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))

    return OLP

def WFNonxyzGrid(grid1,grid2,grid3,Coefficients,Atoms, Basis, cell_vectors=[0.0, 0.0, 0.0], pathtolib=pathtocpp_lib):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms              atoms of the first index
    ##                           list of sublists. 
    ##                           Each of the sublists has five elements. 
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the 
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string. 
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label 
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    
    #Make the grid 
    xyz_grid=np.array(np.meshgrid(grid1,grid2,grid3,indexing="ij")).flatten("F").tolist()
    # Initialize the python lists for Basis Set 1
    atoms_set = []
    positions_set = []
    alphas_lengths_set = []
    alphas_set = []
    contr_coef_set = []
    lms_set = []

    # Create Python lists for input (Set 1)
    for itAtom in range(len(Atoms)):
        Atom_type = Atoms[itAtom][1]
        B = Basis[Atom_type]
        for itBasis in range(len(Basis[Atom_type])):
            atoms_set.append(Atom_type)
            R = np.array(Atoms[itAtom][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R)):
                positions_set.append(R[it1])
            state = B[itBasis]
            dalpha = state[3:]
            alphas_lengths_set.append(len(dalpha))
            for it2 in range(len(dalpha)):
                alphas_set.append(dalpha[it2][0])
                contr_coef_set.append(dalpha[it2][1])
            lm = state[2][:]
            lms_set.append(lm)

    contr_coef_lengths_set= alphas_lengths_set  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_WFN_On_Grid = lib.get_WFN_On_Grid
    get_WFN_On_Grid.restype = POINTER(c_double)
    get_WFN_On_Grid.argtypes = [POINTER(c_double), 
                                c_int,    
                                POINTER(c_double),    
                                POINTER(c_char_p),    
                                POINTER(c_double),
                                POINTER(c_double),
                                POINTER(c_int),
                                POINTER(c_double),
                                POINTER(c_int),
                                POINTER(c_char_p),
                                c_int,
                                POINTER(c_double),
                                c_int]

    freeArray = lib.free_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set_ptr = (c_char_p * len(atoms_set))(*[s.encode("utf-8") for s in atoms_set])
    positions_set_ptr = (c_double * len(positions_set))(*positions_set)
    alphas_set_ptr = (c_double * len(alphas_set))(*alphas_set)
    alphas_lengths_set_ptr = (c_int * len(alphas_lengths_set))(*alphas_lengths_set)
    contr_coef_set_ptr = (c_double * len(contr_coef_set))(*contr_coef_set)
    contr_coef_lengths_set_ptr = (c_int * len(contr_coef_lengths_set))(*contr_coef_lengths_set)
    lms_set_ptr = (c_char_p * len(lms_set))(*[s.encode("utf-8") for s in lms_set])


    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)
    xyzgrid_ptr = (c_double * len(xyz_grid))(*xyz_grid)

    Coefficients_ptr=(c_double * len(Coefficients))(*Coefficients)
    # Call the C++ function
    WFN_On_Grid_array_ptr = get_WFN_On_Grid(xyzgrid_ptr,len(xyz_grid),
                                    Coefficients_ptr,atoms_set_ptr, positions_set_ptr, alphas_set_ptr, alphas_lengths_set_ptr,
                                    contr_coef_set_ptr, contr_coef_lengths_set_ptr, lms_set_ptr, len(atoms_set),
                                    cell_vectors_ptr, len(cell_vectors))
    array_data = np.ctypeslib.as_array(WFN_On_Grid_array_ptr, shape=(int(len(xyz_grid)/3),))
    array_list = deepcopy(array_data)
    freeArray(WFN_On_Grid_array_ptr)

    WFN_on_Grid = np.reshape(np.array(array_list),(len(grid1),len(grid2),len(grid3)),"F")

    return WFN_on_Grid
def LocalPotentialonxyzGrid(gridpoints,MatrixElements,Atoms, Basis, cell_vectors=[0.0, 0.0, 0.0], pathtolib=pathtocpp_lib):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms              atoms of the first index
    ##                           list of sublists. 
    ##                           Each of the sublists has five elements. 
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the 
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string. 
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label 
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    
    #Make the grid 
    xyz_grid=gridpoints
    # Initialize the python lists for Basis Set 1
    atoms_set = []
    positions_set = []
    alphas_lengths_set = []
    alphas_set = []
    contr_coef_set = []
    lms_set = []

    # Create Python lists for input (Set 1)
    for itAtom in range(len(Atoms)):
        Atom_type = Atoms[itAtom][1]
        B = Basis[Atom_type]
        for itBasis in range(len(Basis[Atom_type])):
            atoms_set.append(Atom_type)
            R = np.array(Atoms[itAtom][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R)):
                positions_set.append(R[it1])
            state = B[itBasis]
            dalpha = state[3:]
            alphas_lengths_set.append(len(dalpha))
            for it2 in range(len(dalpha)):
                alphas_set.append(dalpha[it2][0])
                contr_coef_set.append(dalpha[it2][1])
            lm = state[2][:]
            lms_set.append(lm)

    contr_coef_lengths_set= alphas_lengths_set  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_Local_Potential_On_Grid = lib.get_Local_Potential_On_Grid
    get_Local_Potential_On_Grid.restype = POINTER(c_double)
    get_Local_Potential_On_Grid.argtypes = [POINTER(c_double), 
                                c_int,    
                                POINTER(c_double),    
                                POINTER(c_char_p),    
                                POINTER(c_double),
                                POINTER(c_double),
                                POINTER(c_int),
                                POINTER(c_double),
                                POINTER(c_int),
                                POINTER(c_char_p),
                                c_int,
                                POINTER(c_double),
                                c_int]

    freeArray = lib.free_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set_ptr = (c_char_p * len(atoms_set))(*[s.encode("utf-8") for s in atoms_set])
    positions_set_ptr = (c_double * len(positions_set))(*positions_set)
    alphas_set_ptr = (c_double * len(alphas_set))(*alphas_set)
    alphas_lengths_set_ptr = (c_int * len(alphas_lengths_set))(*alphas_lengths_set)
    contr_coef_set_ptr = (c_double * len(contr_coef_set))(*contr_coef_set)
    contr_coef_lengths_set_ptr = (c_int * len(contr_coef_lengths_set))(*contr_coef_lengths_set)
    lms_set_ptr = (c_char_p * len(lms_set))(*[s.encode("utf-8") for s in lms_set])


    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)
    xyzgrid_ptr = (c_double * len(xyz_grid))(*xyz_grid)
    MatrixElementsList=[]
    for it1 in range(np.shape(MatrixElements)[0]):
        for it2 in range(np.shape(MatrixElements)[1]):
            MatrixElementsList.append(MatrixElements[it1][it2])
    MatrixElements_ptr=(c_double * len(MatrixElementsList))(*MatrixElementsList)
    # Call the C++ function
    LocalPotentialonGrid_ptr = get_Local_Potential_On_Grid(xyzgrid_ptr,len(xyz_grid),
                                    MatrixElements_ptr,atoms_set_ptr, positions_set_ptr, alphas_set_ptr, alphas_lengths_set_ptr,
                                    contr_coef_set_ptr, contr_coef_lengths_set_ptr, lms_set_ptr, len(atoms_set),
                                    cell_vectors_ptr, len(cell_vectors))
    array_data = np.ctypeslib.as_array(LocalPotentialonGrid_ptr, shape=(int(len(xyz_grid)/3),))
    array_list = deepcopy(array_data)
    freeArray(LocalPotentialonGrid_ptr)

    LocalPotentialonGrid = np.array(array_list)

    return LocalPotentialonGrid
def get_Position_Operators(Atoms, Basis, cell_vectors=[0.0, 0.0, 0.0], pathtolib=pathtocpp_lib):
    """
    Constructs the position operators (x, y, z) in matrix form for a given 
    set of atoms and basis functions.

    Parameters
    ----------
    Atoms : list
        A list of atoms, where each atom is represented as a list containing:
        - An identifier (=atom index)
        - Atom type (key for looking up basis functions)
        - Cartesian coordinates [x, y, z] in Angstrom units

    Basis : dict
        A dictionary containing basis sets for each atom type.
        Example:
            Basis = {
                'H': [['1', '1', 's', [1, 2.5264751109842596]]],
                'C': [['1', '1', 's', [2, 2.5264751109842596/np.sqrt(0.35355339)]]],
                ...
            }

    cellvectors : list, optional
        A list of three values representing the cell vectors of the simulation cell 
        (for periodic systems). Defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    x_op : numpy.ndarray
        Matrix representation of the x position operator in the atomic basis.

    y_op : numpy.ndarray
        Matrix representation of the y position operator in the atomic basis.

    z_op : numpy.ndarray
        Matrix representation of the z position operator in the atomic basis.

    Notes
    -----
    - Atom positions are converted from Angstroms to atomic units using a conversion factor.
    - Position-dependent transformation matrices are obtained via `getTransformationmatrix()`:
        - modus=0: Overlap-like matrix (OLM)
        - modus=1: x operator base matrix
        - modus=2: y operator base matrix
        - modus=3: z operator base matrix
    - The final position operators are computed as:
        - x_op = x_op_1 + Rx * OLM
        - y_op = y_op_1 + Ry * OLM
        - z_op = z_op_1 + Rz * OLM
    - The function currently returns after processing the first atom’s basis functions.
      This behavior should be reviewed if the intention was to process all atoms.

    Example
    -------
    Atoms = [
        [0, 'H', 0.0, 0.0, 0.0],
        [1, 'H', 0.74, 0.0, 0.0]
    ]
    Basis = {
        'H': ['1s', '2s']
    }
    x_op, y_op, z_op = getPositionOperator(Atoms, Basis)
    """
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    

    # Initialize the python lists for Basis Set 1
    atoms_set1 = []
    positions_set1 = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = np.array(Atoms[itAtom1][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R1)):
                positions_set1.append(R1[it1])
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1

    # Initialize the python lists for Basis Set 2
    atoms_set2 = []
    positions_set2 = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = np.array(Atoms[itAtom2][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R2)):
                positions_set2.append(R2[it1])
            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_Position_Operators = lib.get_Position_Operators
    get_Position_Operators.restype = POINTER(c_double)
    get_Position_Operators.argtypes = [POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_double),
                             c_int,
                             c_int
                             ]

    freeArray = lib.free_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set1_ptr = (c_char_p * len(atoms_set1))(*[s.encode("utf-8") for s in atoms_set1])
    positions_set1_ptr = (c_double * len(positions_set1))(*positions_set1)
    alphas_set1_ptr = (c_double * len(alphas_set1))(*alphas_set1)
    alphas_lengths_set1_ptr = (c_int * len(alphas_lengths_set1))(*alphas_lengths_set1)
    contr_coef_set1_ptr = (c_double * len(contr_coef_set1))(*contr_coef_set1)
    contr_coef_lengths_set1_ptr = (c_int * len(contr_coef_lengths_set1))(*contr_coef_lengths_set1)
    lms_set1_ptr = (c_char_p * len(lms_set1))(*[s.encode("utf-8") for s in lms_set1])

    atoms_set2_ptr = (c_char_p * len(atoms_set2))(*[s.encode("utf-8") for s in atoms_set2])
    positions_set2_ptr = (c_double * len(positions_set2))(*positions_set2)
    alphas_set2_ptr = (c_double * len(alphas_set2))(*alphas_set2)
    alphas_lengths_set2_ptr = (c_int * len(alphas_lengths_set2))(*alphas_lengths_set2)
    contr_coef_set2_ptr = (c_double * len(contr_coef_set2))(*contr_coef_set2)
    contr_coef_lengths_set2_ptr = (c_int * len(contr_coef_lengths_set2))(*contr_coef_lengths_set2)
    lms_set2_ptr = (c_char_p * len(lms_set2))(*[s.encode("utf-8") for s in lms_set2])

    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)

    # Call the C++ function
    OLP_array_ptr = get_Position_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),1)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)

    x_operator = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))
    # Call the C++ function
    OLP_array_ptr = get_Position_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),2)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)
    y_operator = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))
    # Call the C++ function
    OLP_array_ptr = get_Position_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),3)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)
    z_operator = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))

    return x_operator/ConversionFactors['A->a.u.'],y_operator/ConversionFactors['A->a.u.'],z_operator/ConversionFactors['A->a.u.']

def get_momentum_operators(Atoms, Basis, cell_vectors=[0.0, 0.0, 0.0], pathtolib=pathtocpp_lib):
    
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    

    # Initialize the python lists for Basis Set 1
    atoms_set1 = []
    positions_set1 = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = np.array(Atoms[itAtom1][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R1)):
                positions_set1.append(R1[it1])
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1

    # Initialize the python lists for Basis Set 2
    atoms_set2 = []
    positions_set2 = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = np.array(Atoms[itAtom2][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R2)):
                positions_set2.append(R2[it1])
            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_Momentum_Operators = lib.get_Momentum_Operators
    get_Momentum_Operators.restype = POINTER(c_double)
    get_Momentum_Operators.argtypes = [POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_double),
                             c_int,
                             c_int
                             ]

    freeArray = lib.free_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set1_ptr = (c_char_p * len(atoms_set1))(*[s.encode("utf-8") for s in atoms_set1])
    positions_set1_ptr = (c_double * len(positions_set1))(*positions_set1)
    alphas_set1_ptr = (c_double * len(alphas_set1))(*alphas_set1)
    alphas_lengths_set1_ptr = (c_int * len(alphas_lengths_set1))(*alphas_lengths_set1)
    contr_coef_set1_ptr = (c_double * len(contr_coef_set1))(*contr_coef_set1)
    contr_coef_lengths_set1_ptr = (c_int * len(contr_coef_lengths_set1))(*contr_coef_lengths_set1)
    lms_set1_ptr = (c_char_p * len(lms_set1))(*[s.encode("utf-8") for s in lms_set1])

    atoms_set2_ptr = (c_char_p * len(atoms_set2))(*[s.encode("utf-8") for s in atoms_set2])
    positions_set2_ptr = (c_double * len(positions_set2))(*positions_set2)
    alphas_set2_ptr = (c_double * len(alphas_set2))(*alphas_set2)
    alphas_lengths_set2_ptr = (c_int * len(alphas_lengths_set2))(*alphas_lengths_set2)
    contr_coef_set2_ptr = (c_double * len(contr_coef_set2))(*contr_coef_set2)
    contr_coef_lengths_set2_ptr = (c_int * len(contr_coef_lengths_set2))(*contr_coef_lengths_set2)
    lms_set2_ptr = (c_char_p * len(lms_set2))(*[s.encode("utf-8") for s in lms_set2])

    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)

    # Call the C++ function
    OLP_array_ptr = get_Momentum_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),1)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)

    p_x = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))
    # Call the C++ function
    OLP_array_ptr = get_Momentum_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),2)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)
    p_y = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))
    # Call the C++ function
    OLP_array_ptr = get_Momentum_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),3)

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)
    p_z = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))

    return -1.0j*p_x,-1.0j*p_y,-1.0j*p_z
# Define a struct matching std::complex<double>
class ComplexDouble(Structure):
    _fields_ = [
        ("real", c_double),
        ("imag", c_double)
    ]
    def __repr__(self):
        return f"ComplexDouble(real={self.real}, imag={self.imag})"
def get_phase_operators(Atoms, Basis,q_vector=[0.0,0.0,0.0], cell_vectors=[0.0, 0.0, 0.0], cutoff_radius=50, pathtolib=pathtocpp_lib):
    
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    # Initialize the python lists for Basis Set 1
    atoms_set1 = []
    positions_set1 = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = np.array(Atoms[itAtom1][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R1)):
                positions_set1.append(R1[it1])
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1

    # Initialize the python lists for Basis Set 2
    atoms_set2 = []
    positions_set2 = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms)):
        Atom_type2 = Atoms[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = np.array(Atoms[itAtom2][2:]) * ConversionFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R2)):
                positions_set2.append(R2[it1])
            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_Phase_Operators = lib.get_Phase_Operators
    get_Phase_Operators.restype = POINTER(ComplexDouble)
    get_Phase_Operators.argtypes = [POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_double),
                             c_int,
                             POINTER(c_double),
                             c_double
                             ]

    freeArray = lib.free_ptr_complex
    freeArray.argtypes = [POINTER(ComplexDouble)]

    # Convert Python lists to pointers
    atoms_set1_ptr = (c_char_p * len(atoms_set1))(*[s.encode("utf-8") for s in atoms_set1])
    positions_set1_ptr = (c_double * len(positions_set1))(*positions_set1)
    alphas_set1_ptr = (c_double * len(alphas_set1))(*alphas_set1)
    alphas_lengths_set1_ptr = (c_int * len(alphas_lengths_set1))(*alphas_lengths_set1)
    contr_coef_set1_ptr = (c_double * len(contr_coef_set1))(*contr_coef_set1)
    contr_coef_lengths_set1_ptr = (c_int * len(contr_coef_lengths_set1))(*contr_coef_lengths_set1)
    lms_set1_ptr = (c_char_p * len(lms_set1))(*[s.encode("utf-8") for s in lms_set1])

    atoms_set2_ptr = (c_char_p * len(atoms_set2))(*[s.encode("utf-8") for s in atoms_set2])
    positions_set2_ptr = (c_double * len(positions_set2))(*positions_set2)
    alphas_set2_ptr = (c_double * len(alphas_set2))(*alphas_set2)
    alphas_lengths_set2_ptr = (c_int * len(alphas_lengths_set2))(*alphas_lengths_set2)
    contr_coef_set2_ptr = (c_double * len(contr_coef_set2))(*contr_coef_set2)
    contr_coef_lengths_set2_ptr = (c_int * len(contr_coef_lengths_set2))(*contr_coef_lengths_set2)
    lms_set2_ptr = (c_char_p * len(lms_set2))(*[s.encode("utf-8") for s in lms_set2])

    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)

    q_vector_ptr = (c_double * len(q_vector))(*q_vector)

    # Converting the cutoff_radius into atomic units
    cutoff_radius = cutoff_radius * ConversionFactors['A->a.u.']

    # Call the C++ function
    OLP_array_ptr = get_Phase_Operators(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors),q_vector_ptr,cutoff_radius)

    n_elements = len(atoms_set1) * len(atoms_set2)

    # Convert to NumPy array of complex128
    raw_array = np.ctypeslib.as_array(OLP_array_ptr, shape=(n_elements,))
    complex_array = np.array([complex(raw_array[i][0], raw_array[i][1]) for i in range(n_elements)], dtype=np.complex128)

    # Free the memory allocated in C++
    freeArray(OLP_array_ptr)

    phi_q = complex_array.reshape((len(atoms_set1), len(atoms_set2)))
    return phi_q
