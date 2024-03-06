import numpy as np
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import os
import Modules.Util as util 


def writeXSFFile(data,filename="test.xsf",parentfolder='./'):
    '''Function to write a .xsf file that is readable by e.g. Jmol, the origin is assumed to be [0.0,0.0,0.0]
       input:   data:             (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  f                 (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    import datetime
    now = datetime.datetime.now()
    coordinates=Read.readinAtomicCoordinates(parentfolder)
    numofatoms=len(coordinates)
    Nx=np.shape(data)[0]
    Ny=np.shape(data)[1]
    Nz=np.shape(data)[2]
    origin=[0.0,0.0,0.0]
    CellSizes=Read.readinCellSize(parentfolder)
    with open(parentfolder+"/"+filename,'w') as file:
        file.write("#\n")
        file.write("# .xsf file generated with Parabola\n")
        file.write("# "+now.strftime("%Y-%m-%d %H:%M:%S")+"\n")
        file.write("#\n")
        file.write("CRYSTAL\n")
        file.write("PRIMVEC\n")
        file.write(format(CellSizes[0][0],'12.6f')+format(CellSizes[0][1],'12.6f')+format(CellSizes[0][2],'12.6f')+"\n")
        file.write(format(CellSizes[1][0],'12.6f')+format(CellSizes[1][1],'12.6f')+format(CellSizes[1][2],'12.6f')+"\n")
        file.write(format(CellSizes[2][0],'12.6f')+format(CellSizes[2][1],'12.6f')+format(CellSizes[2][2],'12.6f')+"\n")
        file.write("CONVVEC\n")
        file.write(format(CellSizes[0][0],'12.6f')+format(CellSizes[0][1],'12.6f')+format(CellSizes[0][2],'12.6f')+"\n")
        file.write(format(CellSizes[1][0],'12.6f')+format(CellSizes[1][1],'12.6f')+format(CellSizes[1][2],'12.6f')+"\n")
        file.write(format(CellSizes[2][0],'12.6f')+format(CellSizes[2][1],'12.6f')+format(CellSizes[2][2],'12.6f')+"\n")
        file.write("PRIMCOORD\n")
        file.write(format(numofatoms,'5.0f')+format(1,'5.0f')+"\n")
        for atom in coordinates:
            file.write(atom[1]+" "+format(atom[2],'12.6f')+format(atom[3],'12.6f')+format(atom[4],'12.6f')+"\n")
        file.write("BEGIN_BLOCK_DATAGRID_3D\n")
        file.write("3D_field\n")
        file.write("BEGIN_DATAGRID_3D_UNKNOWN\n")
        file.write(format(Nx,'5.0f')+format(Ny,'5.0f')+format(Nz,'5.0f')+"\n")
        file.write(format(origin[0],'5.1f')+format(origin[1],'5.1f')+format(origin[2],'5.1f')+"\n")
        file.write(format(np.linalg.norm(CellSizes[0])*(Nx-1)/Nx,'12.6f')+format(0.0,'12.6f')+format(0.0,'12.6f')+"\n")
        file.write(format(0.0,'12.6f')+format(np.linalg.norm(CellSizes[1])*(Ny-1)/Ny,'12.6f')+format(0.0,'12.6f')+"\n")
        file.write(format(0.0,'12.6f')+format(0.0,'12.6f')+format(np.linalg.norm(CellSizes[2])*(Nz-1)/Nz,'12.6f')+"\n")
        for itz in range(Nz):
            for ity in range(Ny):
                for itx in range(Nx):
                    if (itx or ity or itz) and itx%6==0:
                        file.write("\n")
                    file.write(" {0: .5E}".format(data[itx,ity,itz]))
        file.write("END_DATAGRID_3D\n")
        file.write("END_BLOCK_DATAGRID_3D")

       
    
    return       
    
    
def writeCubeFile(data,filename='test.cube',parentfolder='./'):
    '''Function to write a .cube file that is readable by e.g. Jmol, the origin is assumed to be [0.0,0.0,0.0]
       input:   data:             (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  f                 (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    coordinates=Read.readinAtomicCoordinates(parentfolder)
    numofatoms=len(coordinates)
    conFactors=PhysConst.ConversionFactors()
    Nx=np.shape(data)[0]
    Ny=np.shape(data)[1]
    Nz=np.shape(data)[2]
    origin=[0.0,0.0,0.0]
    CellSizes=Read.readinCellSize(parentfolder)
    with open(parentfolder+"/"+filename,'w') as file:
        file.write("Cube File generated with Parabola\n")
        file.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        file.write(format(numofatoms,'5.0f')+format(origin[0],'12.6f')+format(origin[1],'12.6f')+format(origin[2],'12.6f')+"\n")
        file.write(format(Nx,'5.0f')+format(CellSizes[0][0]*conFactors['A->a.u.']/Nx,'12.6f')+format(CellSizes[0][1]*conFactors['A->a.u.']/Nx,'12.6f')+format(CellSizes[0][2]*conFactors['A->a.u.']/Nx,'12.6f')+"\n")
        file.write(format(Ny,'5.0f')+format(CellSizes[1][0]*conFactors['A->a.u.']/Ny,'12.6f')+format(CellSizes[1][1]*conFactors['A->a.u.']/Ny,'12.6f')+format(CellSizes[1][2]*conFactors['A->a.u.']/Ny,'12.6f')+"\n")
        file.write(format(Nz,'5.0f')+format(CellSizes[2][0]*conFactors['A->a.u.']/Nz,'12.6f')+format(CellSizes[2][1]*conFactors['A->a.u.']/Nz,'12.6f')+format(CellSizes[2][2]*conFactors['A->a.u.']/Nz,'12.6f')+"\n")
        for atom in coordinates:
            file.write(format(PhysConst.AtomSymbolToAtomnumber(atom[1]),'5.0f')+format(0.0,'12.6f')+format(conFactors['A->a.u.']*atom[2],'12.6f')+format(conFactors['A->a.u.']*atom[3],'12.6f')+format(conFactors['A->a.u.']*atom[4],'12.6f')+"\n")
        for itx in range(Nx):
            for ity in range(Ny):
                for itz in range(Nz):
                    if (itx or ity or itz) and itz%6==0:
                        file.write("\n")
                    file.write(" {0: .5E}".format(data[itx,ity,itz]))
    return
    
    
def writemolFile(normalmodeEnergies,normalmodes,normfactors,parentfolder="./"):
    '''
    Function to generate a .mol file for use with e.g., Jmol.
    
    Parameters:
    - normalmodeEnergies (np.array): The normal mode energies as a numpy array.
    - normalmodes (np.array): Normalized Cartesian displacements, i.e., vectors proportional to the Cartesian displacements.
                              v = sqrt(M^(-1)) @ X, where X are the eigenvectors of the rescaled Hessian.
    - normfactors (np.array): The normalization factors n = sqrt(dot(v, v)).
    - parentfolder (str, optional): The parent folder where the .mol file will be created. Default is "./".

    Notes:
    - The function relies on the `ConversionFactors` class.
    - The function looks for a single .xyz file in the specified directory and reads the atomic coordinates from it.
    - The .mol file is generated with sections for frequencies, atomic coordinates, normalization factors, and vibrational modes.
    - The generated .mol file is named "Vibrations.mol" and is saved in the specified parent folder.
    '''
    ConFactor=PhysConst.ConversionFactors()
    xyzfilename=util.getxyzfilename()
    ##########################################
    #Get the number of atoms from the xyz file
    ##########################################
    numofatoms=0
    with open(parentfolder+"/"+xyzfilename) as g:
        lines=g.readlines()
        numofatoms=int(lines[0])
        moldencoordinates=[]
        for line in lines[2:]:
            if len(line.split())>0:
                atom=line.split()[0]
                moldencoordinates.append([atom,float(line.split()[1])*ConFactor['A->a.u.'],float(line.split()[2])*ConFactor['A->a.u.'],float(line.split()[3])*ConFactor['A->a.u.']])
    with open(parentfolder+"/Vibrations.mol",'w') as f:
        f.write('[Molden Format]\n')
        f.write('[FREQ]\n')
        for Frequency in normalmodeEnergies:
            f.write('   '+str(Frequency)+'\n')
        f.write('[FR-COORD]\n')
        for atoms in moldencoordinates:
            f.write(atoms[0]+'   '+str(atoms[1])+'   '+str(atoms[2])+'   '+str(atoms[3])+'\n')
        f.write('[NORM-FACTORS]\n')
        for normfactor in normfactors:
            f.write(str(normfactor)+'\n')
        f.write('[FR-NORM-COORD]\n')
        modeiter=1
        for mode in normalmodes:
            f.write('vibration      '+str(modeiter)+'\n')
            for s in range(numofatoms):
                f.write('   '+str(round(mode[3*s], 12))+'   '+str(round(mode[3*s+1], 12))+'   '+str(round(mode[3*s+2],12))+'\n')
            modeiter+=1

def writexyzfile(atomicsym,coordinates,readpath="./",writepath="./",filename="dummyname",append=False):
    xyzfilename=util.getxyzfilename(readpath)
    xyzfilename=xyzfilename.split("/")[-1]
    # generate new xyz file
    xyzinput=[]
    with open(readpath+"/"+xyzfilename) as g:
        lines = g.readlines()
        xyzinput.append(lines[0])
        xyzinput.append(lines[1])
    g.close()
    for it,xyz in enumerate(coordinates):
        xyzinput.append(atomicsym[it]+' '+str(xyz[0])+' '+str(xyz[1])+' '+str(xyz[2])+'\n')
    #open the old xyz file
    
    #Check if the file exist in the write directory
    inp_files = [f for f in os.listdir(writepath) if f.endswith('.xyz')]
    if len(inp_files) == 0:
        os.system("touch "+xyzfilename)
    if filename=="dummyname":
        filename=xyzfilename
    g=open(writepath+"/"+filename,'a')
    if not append:
        # kill its content
        g.truncate(0)
    #generate the new content
    for line in xyzinput:
        g.write(line)
    g.close()

