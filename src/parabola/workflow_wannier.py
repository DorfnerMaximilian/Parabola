import parabola as p

calcpath='D:\COF_Laura/monolayer/331/'
#calcpath = 'D:\graphene/sup991/'
#calcpath = 'D:/ndiht2/'
mol = p.Molecular_Structure.MolecularStructure(name = 'cofcc', path = calcpath, electronic_path = calcpath)


# parameters to be decided:
nh = 3
nl = 0
nh_plotting_dft = 5
nl_plotting_dft = 0
periodic_copies = [2,2,2]
frags = [[[1,2,3,4],['s','p']],[[1,2,3,4],['s','d']],[[1,2],['p']]]
bandsfilename = 'cofPBE.bs'
nearest_neighbours = 3

#estate_dict = mol.electronic_structure.indexmap['alpha']
#for number in range(-5,5):
#    this_state = estate_dict[number]
#    this_state_en = mol.electronic_structure.ElectronicEigenstates["alpha"][this_state[0]][int(this_state[1])].energy
#    print('energy check: ', number, this_state_en)

# functions to run:
#p.Wannier_final.recommended_kpath_bandstruc(mol,write_flag=True,path=calcpath)
p.Wannier.testing_bloch(mol,path=calcpath)
#with open(calcpath+"allbands.pickle", "rb") as f:
#    allbands = pickle.load(f)

p.Wannier_final.compare_to_dft_bandstruc(mol,nh_plotting_dft,nl_plotting_dft,mode='b',bsfilename=bandsfilename,path=calcpath,band_index_results=allbands)

p.Wannier_final.wannierise(mol,band_index_results=allbands,frags=frags,path=calcpath,Wannier_file_name='Wannier_Orbitals')
p.Wannier_final.general_wan_interpolate_bands(mol, nh_plotting_dft, nl_plotting_dft, bsfilename=bandsfilename, upto_neighbour=nearest_neighbours,Wan_file='Wannier_Orbitals.npy',path=calcpath)

#p.Wannier_final.recommended_kpath_bandstruc(mol,write_flag=True,path=calcpath)

#allbands = p.Wannier_final.band_index(mol,nh,nl,periodic_copies=periodic_copies,path=calcpath)
#with open(calcpath+"allbands.pickle", "wb") as f:
#    pickle.dump(allbands, f)

#with open(calcpath+"allbands.pickle", "rb") as f:
#    allbands = pickle.load(f)

#p.Wannier_final.compare_to_dft_bandstruc(mol,nh_plotting_dft,nl_plotting_dft,mode='b',bsfilename=bandsfilename,path=calcpath,band_index_results=allbands)

#p.Wannier_final.wannierise(mol,band_index_results=allbands,frags=frags,path=calcpath,Wannier_file_name='Wannier_Orbitals')
#p.Wannier_final.general_wan_interpolate_bands(mol, nh_plotting_dft, nl_plotting_dft, bsfilename=bandsfilename, upto_neighbour=nearest_neighbours,Wan_file='Wannier_Orbitals.npy',path=calcpath)
#p.Wannier_final.wan_real_plot(mol, mode='q', periodic_copies=periodic_copies, ind=[0,1,2], Wan_npy='Wannier_Orbitals.npy', path=calcpath)
#p.Wannier_final.wan_real_plot(mol, mode='a', periodic_copies=periodic_copies, ind=[0,1,2], Wan_npy='Wannier_Orbitals.npy', path=calcpath, N1=100, N2=100, N3=100)
