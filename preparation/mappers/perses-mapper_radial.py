import yaml
import numpy as np
from functools import partial
from rdkit import Chem
import openfe
from openfe import SmallMoleculeComponent
from openfe.setup.ligand_network_planning import generate_radial_network
from openfe.setup.atom_mapping.perses_mapper import PersesAtomMapper
from openfe.setup.atom_mapping.perses_scorers import default_perses_scorer
from openmm import unit
import glob

paths = glob.glob('data/*')
paths.remove('data/targets.yml')

for path in paths:
    print(path)
    # load ligands
    ligands_supp = Chem.SDMolSupplier('%s/02_ligands/ligands.sdf'%path, removeHs=False)
    ligands = [SmallMoleculeComponent.from_rdkit(m) for m in ligands_supp]
    
    # Find central ligand
    smallest = None
    for ligand in ligands_supp:
        nr_heavy_atoms = Chem.rdMolDescriptors.CalcNumHeavyAtoms(ligand)
        if smallest is None or nr_heavy_atoms < smallest:
            smallest = nr_heavy_atoms
            central_ligand = SmallMoleculeComponent.from_rdkit(ligand)
            
    # Remove central ligand from ligand list
    ligands = [m for m in ligands if m is not central_ligand]
    
    # create radial graph
    mapper = PersesAtomMapper(allow_ring_breaking=False, preserve_chirality=True,
                              use_positions=True, coordinate_tolerance=0.2*unit.angstrom)
    scorer_fn = partial(default_perses_scorer, use_positions=True, normalize=False)
    
    network = generate_radial_network(ligands=ligands, central_ligand=central_ligand,
                                      mappers=[mapper], scorer=scorer_fn)
    
    # Get edges
    edges = [edge for edge in network.edges]

    # Analyze distance overlap for manual validation
    for i, edge in enumerate(edges):
        if np.any(edge.get_distances() > 1):
            name = f"{edge.componentA.name}_{edge.componentB.name}"
            print(f"{name} exceeds 1 A pairwise distance")

    output = {}
    output['remarks'] = "Radial network graph (star map), central ligand has lowest heavy atom count"
    output['planner'] = "openfe v0.9.2 radial network"

    edges_dict = dict()

    for edge in network.edges:
        edge_name = f'edge_{edge.componentA.name}_{edge.componentB.name}'
        mapping = [list(i) for i in edge.componentA_to_componentB.items()]
        edges_dict[edge_name] = {
            'ligand_a': edge.componentA.name,
            'ligand_b': edge.componentB.name,
            'atom mapping': edge.componentA_to_componentB,
            'score': {
                'value': f"{edge.annotations['score']:.3f}",
                'method': "Perses default geometric scorer", },
            'mapper': "Perses 0.10.1 (allow_ring_breaking=False, preserve_chirality=True, "
                      "use_positions=True, coordinate_tolerance=0.2)",
            'remarks': None,
            }

    output['edges'] = edges_dict

    with open('%s/03_edges/01_perses_star_map_openfe.yml'%path, 'w') as yaml_file:
        yaml.dump(output, yaml_file, default_flow_style=None, sort_keys=False)
