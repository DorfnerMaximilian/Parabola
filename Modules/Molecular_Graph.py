import Modules.Read as Read
import numpy as np
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
class MolecularGraph():
    def __init__(self,path="./"):
        self.path=path
        self.atoms=[]
        self.adjacency_matrix=None
        self.bonds={}
        self.valence_one_targets=[]
        self.number_of_edges=-1
        self.cycles=[]
        self.determine_MolecularGraph_from_xyz()
    def determine_MolecularGraph_from_xyz(self):
        determine_atoms(self)
        ## Determine Bonds
        determine_bonds(self)
        determine_bond_order(self)
        determine_all_cycles(self)
        cycle_vectors=encode_cycles_as_vectors(self)
        sssr_basis = gaussian_elimination_mod2(cycle_vectors)
        decoded_cycles=decode_sssr_vectors(self, sssr_basis)
        self.cycles=decoded_cycles
class atom:
    def __init__(self, symbol, index,coordinates=None):
        self.symbol = symbol
        self.atom_index = index
        self.coordinates=coordinates
        self.valence = octettrules[symbol]
        self.adjacent=[]
    def __repr__(self):
        return f"atom(symbol='{self.symbol}', index={self.atom_index}, valence={self.valence}, adjacent={self.adjacent})"
class bond:
    def __init__(self, atom_index_1, atom_index_2,bond_index,distance=None, order=None):
        self.atom_index_1 = atom_index_1
        self.atom_index_2 = atom_index_2
        self.bond_index=bond_index
        self.bond_order = order
        self.distance=distance
    def __repr__(self):
        return f"bond(index={self.bond_index}, (i,j)={self.atom_index_1,self.atom_index_2}, distance={self.distance}, bond_order={self.bond_order})"
def get_adjacency_matrix(coordinates,atomicsymbols,tol=0.4):
    # Covalent radii dictionary (in angstroms)
    covalent_radii = {
        'H': 0.31,
        'C': 0.76,
        'N': 0.71,
        'O': 0.66,
        'F': 0.57,
        'P': 1.07,
        'S': 1.05,
        # Add more as needed
    }
    n = len(atomicsymbols)
    adj = np.zeros((n, n), dtype=int)
    
    # Compute pairwise distances
    dist_matrix = squareform(pdist(coordinates))
    
    for i in range(n):
        for j in range(i + 1, n):
            r_sum = covalent_radii.get(atomicsymbols[i], 0.7) + covalent_radii.get(atomicsymbols[j], 0.7) + tol
            if dist_matrix[i, j] <= r_sum:
                adj[i, j] = 1
                adj[j, i] = 1  # symmetric
    return adj
def determine_atoms(self):
    coordinates, _, atomic_symbols = Read.readCoordinatesAndMasses(self.path)
    self.adjacency_matrix=get_adjacency_matrix(coordinates,atomic_symbols)
    for it,atom_sym in enumerate(atomic_symbols):
        self.atoms.append(atom(atom_sym,index=it,coordinates=coordinates[it]))
def determine_bonds(self):
    adjacency_matrix=self.adjacency_matrix
    n = len(adjacency_matrix)
    bond_index = 0
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only (undirected graph)
            if adjacency_matrix[i][j]:
                distance=np.linalg.norm(self.atoms[j].coordinates-self.atoms[i].coordinates)
                self.bonds[(i,j)]=bond(i,j,bond_index,distance)
                self.bonds[(j,i)]=bond(j,i,bond_index,distance)
                self.atoms[i].adjacent.append(j)
                self.atoms[j].adjacent.append(i)
                bond_index += 1
    self.number_of_edges=bond_index
def determine_all_cycles(self, max_length=10):
    """
    Finds all simple cycles in the graph up to a maximum length. This corrected
    version preserves the cycle path order needed for edge encoding.
    """
    atoms = self.atoms
    adj_list = [atom.adjacent for atom in atoms]
    n = len(adj_list)
    
    # Use a set to store the frozenset of edges for each unique cycle found
    unique_cycle_edges = set()
    self.cycles = []

    def dfs(path):
        start_node = path[0]
        current_node = path[-1]

        for neighbor in adj_list[current_node]:
            # Avoid backtracking immediately
            if len(path) > 1 and neighbor == path[-2]:
                continue
            
            # If the neighbor closes the cycle
            if neighbor == start_node and len(path) > 2:
                # Create a canonical representation of the cycle's edges
                # (a frozenset of sorted tuples) to check for uniqueness.
                edges = frozenset(tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)) | \
                        frozenset({tuple(sorted((path[-1], path[0])))})

                if edges not in unique_cycle_edges:
                    unique_cycle_edges.add(edges)
                    self.cycles.append(path) # Store the actual path with correct order
                continue # Continue search from current_node for other cycles
            
            # Continue DFS if neighbor is not visited and path is not too long
            if neighbor not in path and len(path) < max_length:
                dfs(path + [neighbor])

    # Start a DFS from each node
    for node in range(n):
        dfs([node])



def encode_cycles_as_vectors(self):
    """
    Encodes a cycle as a binary vector over edges.

    Args:
        cycle (list[int]): List of node indices forming a cycle.
        edge_indices (dict): Mapping from edge (i, j) to index.
        num_edges (int): Total number of edges in the graph.

    Returns:
        vec (np.ndarray): Binary vector indicating which edges are in the cycle.
    """
    num_edges=self.number_of_edges
    cycles=self.cycles
    vector_encoding=[]
    for cycle in cycles:
        vec = np.zeros(num_edges, dtype=int)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]  # wrap around to form a cycle
            idx = self.bonds[(u, v)].bond_index
            if idx is not None:
                vec[idx] = 1
        vector_encoding.append(vec)
    return vector_encoding


def gaussian_elimination_mod2(vectors):
    """
    Performs fast, matrix-based Gaussian elimination on binary vectors (modulo 2).

    Args:
        vectors (list[np.ndarray]): List of binary vectors (0 or 1 values).

    Returns:
        basis (list[np.ndarray]): A linearly independent basis in row-echelon form.
    """
    if not vectors:
        return []
    
    # 1. Stack vectors into a matrix for efficient processing.
    # We use uint8 for memory efficiency with binary data.
    matrix = np.array(vectors, dtype=np.uint8)

    # The heuristic to sort by vector weight can be preserved if needed.
    # It may not always improve speed but matches the original logic.
    # row_sums = np.sum(matrix, axis=1)
    # sorted_indices = np.argsort(row_sums)
    # matrix = matrix[sorted_indices]
    
    # 2. Perform Gaussian elimination in-place.
    num_rows, num_cols = matrix.shape
    pivot_row = 0
    # Iterate through columns to find pivots
    for j in range(num_cols):
        # Stop if all remaining rows are zero
        if pivot_row >= num_rows:
            break

        # Find a row with a 1 in the current pivot column
        i = pivot_row
        while i < num_rows and matrix[i, j] == 0:
            i += 1

        if i < num_rows:
            # Swap this row to the pivot position
            matrix[[pivot_row, i]] = matrix[[i, pivot_row]]
            
            # Eliminate other 1s in this column by XORing with the pivot row.
            # This is a highly vectorized operation.
            rows_to_xor = np.where(matrix[:, j] == 1)[0]
            # Exclude the pivot row itself from the XOR operation
            rows_to_xor = np.setdiff1d(rows_to_xor, pivot_row, assume_unique=True)
            
            if rows_to_xor.size > 0:
                matrix[rows_to_xor] = (matrix[rows_to_xor] + matrix[pivot_row]) % 2

            pivot_row += 1
            
    # 3. The non-zero rows of the reduced matrix form the basis.
    # Find the rank by counting non-zero rows.
    rank = np.sum(np.any(matrix, axis=1))
    basis_matrix = matrix[:rank]
    
    # Convert back to a list of arrays to match the original output format.
    return [row for row in basis_matrix]
def decode_sssr_vectors(self, sssr_basis):
    """
    Decodes the binary vectors of the SSSR basis back into cycle paths.

    Args:
        sssr_basis (list[np.ndarray]): The basis vectors from Gaussian elimination.

    Returns:
        decoded_cycles (list[list[int]]): A list of cycles, where each cycle
                                           is a list of atom indices in path order.
    """
    # 1. Create a reverse mapping from bond_index to the edge tuple (u, v).
    # This is the crucial step to go from an index back to atoms.
    index_to_edge = {}
    for edge_tuple, bond_obj in self.bonds.items():
        # Ensure each bond index is added only once
        if bond_obj.bond_index not in index_to_edge:
            index_to_edge[bond_obj.bond_index] = (bond_obj.atom_index_1, bond_obj.atom_index_2)
    
    decoded_cycles = []
    for vec in sssr_basis:
        # 2. Find the indices of edges that form the cycle.
        edge_indices = np.where(vec == 1)[0]
        if len(edge_indices) < 3:  # A cycle must have at least 3 edges.
            continue
        
        # 3. Build a temporary adjacency list for this specific cycle.
        cycle_adj = {}
        for idx in edge_indices:
            u, v = index_to_edge[idx]
            if u not in cycle_adj: cycle_adj[u] = []
            if v not in cycle_adj: cycle_adj[v] = []
            cycle_adj[u].append(v)
            cycle_adj[v].append(u)
        
        # 4. Reconstruct the path by walking along the edges.
        # Every node in a simple cycle has exactly two neighbors.
        start_node = next(iter(cycle_adj)) # Pick an arbitrary starting node
        path = [start_node]
        prev_node = None
        current_node = start_node
        
        # Walk along the cycle until it's fully traversed
        for _ in range(len(cycle_adj) - 1):
            neighbors = cycle_adj[current_node]
            # Find the next node that is not the one we just came from
            next_node = neighbors[0] if neighbors[0] != prev_node else neighbors[1]
            path.append(next_node)
            prev_node = current_node
            current_node = next_node
            
        decoded_cycles.append(path)
        
    return decoded_cycles
octettrules = {
        'H': 1,
        'C': 4,
        'N': 3,
        'O': 2,
        'F': 1,
        'P': 3,
        'S': 2,
        # Add more as needed
    }
def determine_bond_order(self):
    for atom in self.atoms:
        if len(atom.adjacent)==1:
            atomtype=atom.symbol
            idx=atom.atom_index
            self.valence_one_targets.append(idx)
            idx_ad=atom.adjacent[0]
            self.atoms[idx_ad].valence-=octettrules[atomtype]
            atom.valence=0
            self.bonds[(idx,idx_ad)].bond_order=octettrules[atomtype]
            self.bonds[(idx_ad,idx)].bond_order=octettrules[atomtype]

    for atom in self.atoms:
        if len(atom.adjacent)==2:
            idx=atom.atom_index
            atomtype=atom.symbol
            idx_ad_1=atom.adjacent[0]
            idx_ad_2=atom.adjacent[1]
            if type(self.bonds[(idx,idx_ad_1)].bond_order) != None and type(self.bonds[(idx,idx_ad_2)].bond_order) == None:
                self.bonds[(idx,idx_ad_2)].bond_order=octettrules[atomtype]-self.bonds[(idx,idx_ad_1)].bond_order
                self.bonds[(idx_ad_2,idx)].bond_order=octettrules[atomtype]-self.bonds[(idx,idx_ad_1)].bond_order
            elif type(self.bonds[(idx,idx_ad_2)].bond_order) != None and type(self.bonds[(idx,idx_ad_1)].bond_order) == None:
                self.bonds[(idx,idx_ad_1)].bond_order=octettrules[atomtype]-self.bonds[(idx,idx_ad_2)].bond_order
                self.bonds[(idx_ad_1,idx)].bond_order=octettrules[atomtype]-self.bonds[(idx,idx_ad_2)].bond_order
            else:
                rem_valence = atom.valence
                if rem_valence > 0 and rem_valence % 2 == 0:
                    bond_order_to_assign = rem_valence // 2
                    
                    # Assign order to the first bond and update neighbor
                    self.bonds[(idx, idx_ad_1)].bond_order = bond_order_to_assign
                    self.bonds[(idx_ad_1, idx)].bond_order = bond_order_to_assign
                    self.atoms[idx_ad_1].valence -= bond_order_to_assign
                    
                    # Assign order to the second bond and update neighbor
                    self.bonds[(idx, idx_ad_2)].bond_order = bond_order_to_assign
                    self.bonds[(idx_ad_2, idx)].bond_order = bond_order_to_assign
                    self.atoms[idx_ad_2].valence -= bond_order_to_assign
                    atom.valence = 0
                elif rem_valence > 0:
                    if self.atoms[idx_ad_1].symbol==self.atoms[idx_ad_2].symbol:
                        bond_order_to_assign = rem_valence / 2.0
                        # Assign order to the first bond and update neighbor
                        self.bonds[(idx, idx_ad_1)].bond_order = bond_order_to_assign
                        self.bonds[(idx_ad_1, idx)].bond_order = bond_order_to_assign
                        self.atoms[idx_ad_1].valence -= bond_order_to_assign
                        
                        # Assign order to the second bond and update neighbor
                        self.bonds[(idx, idx_ad_2)].bond_order = bond_order_to_assign
                        self.bonds[(idx_ad_2, idx)].bond_order = bond_order_to_assign
                        self.atoms[idx_ad_2].valence -= bond_order_to_assign
                        atom.valence = 0
                    else:
                        # Odd remaining valence is chemically unusual for a degree-2 atom.
                        raise ValueError(f"Cannot determine bond order for atom {idx} ({atom.symbol}): "
                                        f"Odd remaining valence ({rem_valence}).")
    for atom in self.atoms:
        if len(atom.adjacent) == 3:
            idx = atom.atom_index
            atomtype = atom.symbol
            neighbors = atom.adjacent
            known_bonds = []
            unknown_bonds = []
            total_known_order = 0
            
            for neighbor in neighbors:
                bond_key = (idx, neighbor)
                bond_order = getattr(self.bonds[bond_key], 'bond_order', None)
                if bond_order is not None:
                    known_bonds.append((neighbor, bond_order))
                    total_known_order += bond_order
                else:
                    unknown_bonds.append(neighbor)
            rem_valence = octettrules[atomtype] - total_known_order

            if rem_valence < 0:
                raise ValueError(f"Atom {idx} ({atomtype}) is overbonded: assigned more than allowed valence.")
            
            if len(unknown_bonds) == 0:
                # All bond orders are already known
                atom.valence = 0
            elif len(unknown_bonds) == 1:
                # Only one bond order missing, assign remaining valence
                neighbor = unknown_bonds[0]
                self.bonds[(idx, neighbor)].bond_order = rem_valence
                self.bonds[(neighbor, idx)].bond_order = rem_valence
                self.atoms[neighbor].valence -= rem_valence
                atom.valence = 0
            elif len(unknown_bonds) == 2:
                if rem_valence % 2 == 0:
                    bond_order_to_assign = rem_valence // 2
                    for neighbor in unknown_bonds:
                        self.bonds[(idx, neighbor)].bond_order = bond_order_to_assign
                        self.bonds[(neighbor, idx)].bond_order = bond_order_to_assign
                        self.atoms[neighbor].valence -= bond_order_to_assign
                    atom.valence = 0
                elif self.atoms[unknown_bonds[0]].symbol==self.atoms[unknown_bonds[1]].symbol:
                    bond_order_to_assign = rem_valence / 2.0
                    # Assign order to the first bond and update neighbor
                    self.bonds[(idx, unknown_bonds[0])].bond_order = bond_order_to_assign
                    self.bonds[(unknown_bonds[0], idx)].bond_order = bond_order_to_assign
                    self.atoms[unknown_bonds[0]].valence -= bond_order_to_assign
                    
                    # Assign order to the second bond and update neighbor
                    self.bonds[(idx, unknown_bonds[1])].bond_order = bond_order_to_assign
                    self.bonds[(unknown_bonds[1], idx)].bond_order = bond_order_to_assign
                    self.atoms[unknown_bonds[1]].valence -= bond_order_to_assign
                    atom.valence = 0
                else:
                    raise ValueError(f"Cannot evenly assign bond orders to two unknown bonds for atom {idx} ({atomtype}).")
            elif len(unknown_bonds) == 3:
                if rem_valence % 3 == 0:
                    bond_order_to_assign = rem_valence // 3
                    for neighbor in unknown_bonds:
                        self.bonds[(idx, neighbor)].bond_order = bond_order_to_assign
                        self.bonds[(neighbor, idx)].bond_order = bond_order_to_assign
                        self.atoms[neighbor].valence -= bond_order_to_assign
                    atom.valence = 0
                else:
                    raise ValueError(f"Cannot evenly assign bond orders to three unknown bonds for atom {idx} ({atomtype}).")
    

def valence_one_modifications(original_graph: MolecularGraph,type="F",target_index=0) -> MolecularGraph:
    # Step 1: Make a deep copy to avoid mutating the original graph
    graph = deepcopy(original_graph)
    
    
    

    

    # Step 3: Return the modified graph
    return graph


