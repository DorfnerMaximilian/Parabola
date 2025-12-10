import parabola as p

calcpath = "/Users/laura/Desktop/MA/wannier/graph/hex/551/"
mol = p.structure.Molecular_Structure(name="551", path=calcpath, electronics_path=calcpath)


# parameters to be decided:
nh = 3
nl = 0
nh_plotting_dft = 5
nl_plotting_dft = 0
periodic_copies = [2, 2, 2]
frags = [[[1, 2, 3, 4], ["s", "p"]], [[1, 2, 3, 4], ["s", "d"]], [[1, 2], ["p"]]]
bandsfilename = "hex.bs"
nearest_neighbours = 3

# estate_dict = molecular_structure.Electronics.indexmap["alpha"]
# for number in range(-5, 5):
#    this_state = estate_dict[number]
#    this_state_en = molecular_structure.Electronics.energies["alpha"][this_state[0]][int(this_state[1])].energy   ## this is not working properly
#    print("energy check: ", number, this_state_en)

# functions to run:
p.Wannier_final.recommended_kpath_bandstruc(mol, write_flag=True, path=calcpath)
p.Wannier.testing_bloch(mol, path=calcpath)

# allbands = p.Wannier_final.band_index(molecular_structure, nh, nl, periodic_copies=periodic_copies, path=calcpath)
# with open(calcpath + "allbands.pickle", "wb") as f:
#    pickle.dump(allbands, f)
#
## with open(calcpath + "allban1ds.pickle", "rb") as f:
##    allbands = pickle.load(f)
#
# p.Wannier_final.compare_to_dft_bandstruc(
#    molecular_structure,
#    nh_plotting_dft,
#    nl_plotting_dft,
#    mode="b",
#    bsfilename=bandsfilename,
#    path=calcpath,
#    band_index_results=allbands,
# )
#
# p.Wannier_final.wannierise(
#    molecular_structure, band_index_results=allbands, frags=frags, path=calcpath, Wannier_file_name="Wannier_Orbitals"
# )
# p.Wannier_final.general_wan_interpolate_bands(
#    molecular_structure,
#    nh_plotting_dft,
#    nl_plotting_dft,
#    bsfilename=bandsfilename,
#    upto_neighbour=nearest_neighbours,
#    Wan_file="Wannier_Orbitals.npy",
#    path=calcpath,
# )
#
## p.Wannier_final.recommended_kpath_bandstruc(molecular_structure,write_flag=True,path=calcpath)
#
## allbands = p.Wannier_final.band_index(molecular_structure,nh,nl,periodic_copies=periodic_copies,path=calcpath)
## with open(calcpath+"allbands.pickle", "wb") as f:
##    pickle.dump(allbands, f)
#
## with open(calcpath+"allbands.pickle", "rb") as f:
##    allbands = pickle.load(f)
#
## p.Wannier_final.compare_to_dft_bandstruc(molecular_structure,nh_plotting_dft,nl_plotting_dft,mode='b',bsfilename=bandsfilename,path=calcpath,band_index_results=allbands)
#
## p.Wannier_final.wannierise(molecular_structure,band_index_results=allbands,frags=frags,path=calcpath,Wannier_file_name='Wannier_Orbitals')
## p.Wannier_final.general_wan_interpolate_bands(molecular_structure, nh_plotting_dft, nl_plotting_dft, bsfilename=bandsfilename, upto_neighbour=nearest_neighbours,Wan_file='Wannier_Orbitals.npy',path=calcpath)
## p.Wannier_final.wan_real_plot(molecular_structure, mode='q', periodic_copies=periodic_copies, ind=[0,1,2], Wan_npy='Wannier_Orbitals.npy', path=calcpath)
## p.Wannier_final.wan_real_plot(molecular_structure, mode='a', periodic_copies=periodic_copies, ind=[0,1,2], Wan_npy='Wannier_Orbitals.npy', path=calcpath, N1=100, N2=100, N3=100)
#
