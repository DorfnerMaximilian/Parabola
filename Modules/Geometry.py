import Modules.Util as util
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.Write as Write
import numpy as np
import os
def getCoordinatesAndMasses(pathtoxyz="./Equilibrium_Geometry/"):
    """
    Helper routine to parse coordinates and masses from an xyz file.

    Parameters:
    - pathtoxyz (str, optional): Path to the corresponding xyz file where the data is saved.

    Returns:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, in the same ordering as the coordinates.
    - atomicsymbols (list): Atomic symbols corresponding to each atom in the same ordering as the coordinates.

    Notes:
    - This function relies on the `getAtomicCoordinates` function and the `StandardAtomicWeights` functions defined in ReadFile & PhysConst.
    - The `getAtomicCoordinates` function is assumed to return a list of atomic coordinates in the xyz file format.
    - The `StandardAtomicWeights` class is assumed to provide standard atomic weights for element symbols.
    - The returned coordinates are in the format of a numpy array with shape (number of atoms, 3).
    - The returned masses are in the format of a numpy array with shape (number of atoms,).
    - The returned atomicsymbols is a list of atomic symbols in the same order as the coordinates.
    """
    SaW=PhysConst.StandardAtomicWeights()
    #Get the Atomic coordinates 
    AtomicCoordinates=Read.readinAtomicCoordinates(pathtoxyz)
    #Parse the Coordinates & the Masses
    coordinates=[]
    masses=[]
    atomicsymbols=[]
    for coords in AtomicCoordinates:
        mass=SaW[coords[1]]
        atomicsymbols.append(coords[1])
        masses.append(mass)
        coordinates.append(np.array([coords[2],coords[3],coords[4]]))
    return coordinates,masses,atomicsymbols

def ComputeCenterOfMassCoordinates(coordinates,masses):
    """
    Computes the center of mass of a molecule with respect to the basis of the coordinate file.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis (arbitrary origin).
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - centerofmasscoordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis, origin at the center of mass.
    - centerofmass (3x1 numpy.array): Coordinates of the center of mass with respect to the old frame.

    Notes:
    - The coordinates provided have an arbitrary origin.
    - The returned centerofmasscoordinates have their origin at the center of mass.
    - The returned centerofmass is the coordinates of the center of mass with respect to the old frame.
    - The center of mass is computed based on the provided masses and coordinates.
    """
    MassofMolecule=0.0
    centerofmass=np.array([0.,0.,0.])
    #Compute the center of mass: 
    for it,coords in enumerate(coordinates):
        mass=masses[it]
        MassofMolecule+=mass
        centerofmass+=mass*np.array([coords[0],coords[1],coords[2]])
    centerofmass/=MassofMolecule
    
    centerofmasscoordinates=[]
    #get the center of mass coordinates
    for coords in coordinates:
        centerofmasscoordinates.append(coords-centerofmass)
    return centerofmasscoordinates,centerofmass

def getInertiaTensor(coordinates,masses):
    """
    Computes the inertia tensor of a molecule with respect to the basis of the coordinate file.
    Note that the inertia tensor is only a physically meaningful object if the coordinates have
    the center of mass as an origin.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - I (3x3 numpy.array): Moment of inertia tensor of the molecule.

    Notes:
    - This function relies on the `ComputeCenterOfMassCoordinates` function.
    - The coordinates provided should have the center of mass as the origin.
    - The returned inertia tensor is a 3x3 numpy array.
    - The moments of inertia are computed along the principal axes x, y, and z.
    """

    #Compute the coordinates in the frame where the origin is the center of mass
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(coordinates,masses)
    #get the moments of inertia tensor
    Ixx=0.0
    Iyy=0.0
    Izz=0.0
    Ixy=0.0
    Ixz=0.0
    Iyz=0.0
    for it,coords in enumerate(centerofmasscoordinates):
        Ixx+=masses[it]*(coords[1]**2+coords[2]**2)
        Iyy+=masses[it]*(coords[0]**2+coords[2]**2)
        Izz+=masses[it]*(coords[0]**2+coords[1]**2)
        Ixy+=(-1)*masses[it]*coords[0]*coords[1]
        Ixz+=(-1)*masses[it]*coords[0]*coords[2]
        Iyz+=(-1)*masses[it]*coords[1]*coords[2]
    #The moment of inertia tensor
    I=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
    return I
def getPrincipleAxisCoordinates(path="./"):
    ## Function to center the molecule in the unit cell & align the principle Axis 
    ## and orient priciple axis along x y z
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)
    cellcoordinates=Read.readinCellSize(path)
    #Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct =np.abs(np.dot(cellvector1,cellvector2))
            abscrossproduct = np.linalg.norm((np.cross(cellvector1,cellvector2)))
            if absscalarproduct >10**(-10) and abscrossproduct>10**(-10):
                ValueError("Centering of Molecule in Unit Cell makes only sense for Non-Periodic Calculations! Use Orthorhombic Unit cells for this case!")

    #Compute center of Cell (assuming orthogonal basis vectors)
    xyzcoordinates,masses,atomicsym=getCoordinatesAndMasses(path)


    #Compute the Intertia Tensor
    I=getInertiaTensor(xyzcoordinates,masses)
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(xyzcoordinates,masses)
    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    #Check that the principle axis are orthorgonal (Could be not the case in case of degeneracy)
    for it1 in range(3):
        for it2 in range(3):
            if it1!=it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2]))>10**(-12):
                ValueError("Principle Axis of Molecules are not orthorgonal! Implement the Degenerate Case!")
            if it1==it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2])-1.0)>10**(-12):
                ValueError("Principle Axis are not normalized! This leads to Errorious results!")
    #Represent the centerofMassCoordinates in terms of the principle axis frame
    #Fix Orientation of Principle Axis 
    v1=principleAxis[:,0]
    v2=principleAxis[:,1]
    v3=principleAxis[:,2]
    if v1[0]<0.0:
        v1*=-1.0
    if v2[0]<0.0:
        v2*=-1.0
    #This makes Principle Axis righthanded
    B=np.zeros((3,3))
    B[:,0]=v1
    B[:,1]=v2
    B[:,2]=v3
    determinant=np.linalg.det(B)
    if determinant<0:
        v3*=-1.0
    principleaxiscoordinates=[]
    for coordinate in centerofmasscoordinates:
        v1coordinate=np.dot(coordinate,v1)
        v2coordinate=np.dot(coordinate,v2)
        v3coordinate=np.dot(coordinate,v3)
        principleaxiscoordinates.append(np.array([v1coordinate,v2coordinate,v3coordinate]))
    
    return principleaxiscoordinates

def getNeibouringCellVectors(path,m=1):
    ConFactors=PhysConst.ConversionFactors()
    cellvectors=[]
    cell=Read.readinCellSize(path)
    for itx in range(-m,m+1,1):
        for ity in range(-m,m+1,1):
            for itz in range(-m,m+1,1):
                vectortoappend=(itx*cell[0]+ity*cell[1]+itz*cell[2])*ConFactors["A->a.u."]
                cellvectors.append(vectortoappend[0])
                cellvectors.append(vectortoappend[1])
                cellvectors.append(vectortoappend[2])
    return cellvectors
def centerMolecule(path="./"):
    ## Function to center the molecule/collection of atoms in the unit cell 
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    cellcoordinates=Read.readinCellSize(path)
    print(cellcoordinates)
    #Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct =np.abs(np.dot(cellvector1,cellvector2))
            abscrossproduct = np.linalg.norm((np.cross(cellvector1,cellvector2)))
            if absscalarproduct >10**(-10) and abscrossproduct>10**(-10):
                ValueError("Centering of Molecule in Unit Cell makes only sense for Non-Periodic Calculations! Use Orthorhombic Unit cells for this case!")

    #Compute center of Cell (assuming orthogonal basis vectors)
    cellcenter=0.5*cellcoordinates[0]+0.5*cellcoordinates[1]+0.5*cellcoordinates[2]
    xyzcoordinates,_,atomicsym=getCoordinatesAndMasses(path)
    geometric_mean=np.array([0.0,0.0,0.0])
    for coordinate in xyzcoordinates:
        geometric_mean+=coordinate
    geometric_mean/=len(xyzcoordinates)
    centerofcellCoordinates=[]
    for coordinate in xyzcoordinates:
        centerofcellCoordinates.append(coordinate+cellcenter-geometric_mean)
    #Compute maximum distance of atom from coordinatecenter
    maxdistx=0
    maxdisty=0
    maxdistz=0
    for coordinates in xyzcoordinates:
        distx=np.abs(coordinates[0])
        disty=np.abs(coordinates[1])
        distz=np.abs(coordinates[2])
        if distx > maxdistx:
            maxdistx=distx
        if disty > maxdisty:
            maxdisty=disty
        if distz > maxdistz:
            maxdistz=distz
    #Add cutoff to maximum distance
    cutoff=2
    mincellsizex=cutoff+maxdistx
    mincellsizey=cutoff+maxdisty
    mincellsizez=cutoff+maxdistz
    #Check if molecule fits into box otherwise shift coordinates to center of cell
    
    if mincellsizex<=0.5*distx:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    if mincellsizey<=0.5*disty:
        ValueError("Increase cell size in y direction to at least ",2*mincellsizey)
    if mincellsizez<=0.5*distz:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    
    Write.writexyzfile(atomicsym,centerofcellCoordinates,path)
def changeConfiguration(folderlabel,vector,delta,sign,path_xyz='./',path_to="./"):
    ## Reads in the equilibrium configuration xyz file and changes 
    ## position of the atoms in the xyz file in direction of the unit vector vector
    ## either:
    ## by delta [units of corresponding xyzfile or ], if the rescaleflag is False
    ##
    ## input:   folderlabel name of the folder to create                     (string)
    ##          vector      normalized vector of the size of 3 x atoms       (np.array)
    ##          delta       change of the atomic configuration               (float)
    ##          sign                                                         (0,1)
    ## (opt.)   rescaleflag flag to control the rescaling of delta           (bool)
    ## (opt.)   path   path to the folder of the VibAna calculation          (string)
    ## output:  -               (void)

    #Get the .xyz file
    xyz_files = [f for f in os.listdir(path_xyz) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename = xyz_files[0]
    xyzcoordinates=[]
    atomtypes=[]
    with open(path_xyz+"/"+xyzfilename) as g:
        lines = g.readlines()
        for l in lines[2:]:
            if len(l.split())>=4:
                xyzcoordinates.append(float(l.split()[1]))
                xyzcoordinates.append(float(l.split()[2]))
                xyzcoordinates.append(float(l.split()[3]))
                atomtypes.append(l.split()[0])
    xyzcoordinates=np.array(xyzcoordinates)
    xyzcoordinates=xyzcoordinates+(-1)**(sign)*delta*vector
    if sign==0:
        symbolsign='+'
    if sign==1:
        symbolsign='-'
    foldername=path_to+folderlabel+"sign="+symbolsign
    os.mkdir(foldername)
    f=open(foldername+"/"+xyzfilename,"w")
    f.write(lines[0])
    f.write(lines[1])
    for iter in range(int(len(xyzcoordinates)/3)):
        atom=atomtypes[iter]
        xcoord=xyzcoordinates[3*iter]
        ycoord=xyzcoordinates[3*iter+1]
        zcoord=xyzcoordinates[3*iter+2]
        f.write(atom+' '+str(xcoord)+' '+str(ycoord)+' '+str(zcoord))
        f.write("\n")
    f.close()
    return
def changeConfiguration2(folderlabel,vector1,delta1,sign1,vector2,delta2,sign2,path_xyz='./',path_to="./"):
    ## Reads in the equilibrium configuration xyz file and changes 
    ## position of the atoms in the xyz file in direction of the unit vector delta 1 * vector1 +delta2*vector2
    ## either:
    ## by delta [units of corresponding xyzfile or ]
    ##
    ## input:   folderlabel name of the folder to create                     (string)
    ##          vector      normalized vector of the size of 3 x atoms       (np.array)
    ##          delta       change of the atomic configuration               (float)
    ##          sign                                                         (0,1)
    ## (opt.)   rescaleflag flag to control the rescaling of delta           (bool)
    ## (opt.)   path   path to the folder of the VibAna calculation          (string)
    ## output:  -               (void)

    #Get the .xyz file
    xyz_files = [f for f in os.listdir(path_xyz) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename = xyz_files[0]
    xyzcoordinates=[]
    atomtypes=[]
    with open(path_xyz+"/"+xyzfilename) as g:
        lines = g.readlines()
        for l in lines[2:]:
            if len(l.split())>=4:
                xyzcoordinates.append(float(l.split()[1]))
                xyzcoordinates.append(float(l.split()[2]))
                xyzcoordinates.append(float(l.split()[3]))
                atomtypes.append(l.split()[0])
    xyzcoordinates=np.array(xyzcoordinates)
    xyzcoordinates=xyzcoordinates+(-1)**(sign1)*delta1*vector1+(-1)**(sign2)*delta2*vector2
    if sign1==0:
        symbolsign1='+'
    if sign1==1:
        symbolsign1='-'
    if sign2==0:
        symbolsign2='+'
    if sign2==1:
        symbolsign2='-'
    foldername=path_to+folderlabel+"sign1="+symbolsign1+"sign2="+symbolsign2
    os.mkdir(foldername)
    f=open(foldername+"/"+xyzfilename,"w")
    f.write(lines[0])
    f.write(lines[1])
    for iter in range(int(len(xyzcoordinates)/3)):
        atom=atomtypes[iter]
        xcoord=xyzcoordinates[3*iter]
        ycoord=xyzcoordinates[3*iter+1]
        zcoord=xyzcoordinates[3*iter+2]
        f.write(atom+' '+str(xcoord)+' '+str(ycoord)+' '+str(zcoord))
        f.write("\n")
    f.close()
    return
def getNewXYZ(path='.'):
    ## Script to readout the last iteration of the GeoOpt file and 
    ## generate new oldxyzfile_opt.xyz file for Vibrational Analysis
    ## input: (opt.)   path   path to the folder of the geoOpt calculation         (string)
    ## output:  -                                                                  (void)


    #get the Projectname
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = path+"/"+inp_files[0]
    f = open(filename, "r")
    projectname='NoName'
    for line in f.readlines():
        if len(line.split())>0:
            if line.split()[0]=="PROJECT":
                projectname=line.split()[1]
    f.close()
    xyz_files = [f for f in os.listdir(path) if f.endswith('.xyz')]
    #remove the projectname+'-pos-1.xyz' file from xyz files
    xyz_files.remove(projectname+'-pos-1.xyz')
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one old xyz file in the current directory')
    xyzfilename=path+"/"+xyz_files[0]
    iter2=0
    if projectname!='NoName':
        numiter=0
        buffer=[]
        getflag=False
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if len(line.split())>0:
                if line.split()[0]=='i':
                    numiter+=1
        f.close()
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if getflag:
                buffer.append(line.split())
            if len(line.split())>0:
                if line.split()[0]=='i':
                    iter2+=1
                if iter2==numiter:
                    getflag=True
            else:
                continue
        f.close()
        f=open(xyzfilename,'r')
        header=f.readlines()[0:2]
        f.close()
        g=open(xyzfilename[:-4]+'_opt.xyz','w')
        g.write(header[0])
        g.write(header[1])
        for line in buffer:
            for element in line:
                g.write(element)
                g.write(' ')
            g.write('\n')
        g.close()
