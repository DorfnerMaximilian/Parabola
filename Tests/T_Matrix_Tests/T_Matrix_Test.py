import unittest
import Parabola as p
import numpy as np

class TestTMatrix(unittest.TestCase):
    def test_OBC(self):
        print("Started OBC Test")
        OLM_CP2K=np.load("./OBC/OLM_OBC.npy")
        Atoms=p.Read.readinAtomicCoordinates("./OBC/")
        Basis=p.AtomicBasis.getBasis("./OBC/")
        OLM_Parabola=p.AtomicBasis.getTransformationmatrix(Atoms,Atoms,Basis)
        np.testing.assert_array_almost_equal(OLM_Parabola, OLM_CP2K,13,"OBC Test failed!")
    def test_PBC(self):
        print("\nStarted PBC Test")
        OLM_CP2K=np.load("./PBC/OLM_PBC.npy")
        Atoms=p.Read.readinAtomicCoordinates("./PBC/")
        Basis=p.AtomicBasis.getBasis("./PBC/")
        cellvectors=p.Geometry.getNeibouringCellVectors("./PBC/",2,2,2)
        OLM_Parabola=p.AtomicBasis.getTransformationmatrix(Atoms,Atoms,Basis,cellvectors)
        np.testing.assert_array_almost_equal(OLM_Parabola, OLM_CP2K,13,"PBC Test failed!")
	
if __name__ == '__main__':
    unittest.main()

