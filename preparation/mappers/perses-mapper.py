import yaml
import numpy as np
from functools import partial
from rdkit import Chem
from openfe import SmallMoleculeComponent
from openfe.setup.ligand_network_planning import generate_minimal_spanning_network
from openfe.setup.atom_mapping.perses_mapper import PersesAtomMapper
from openfe.setup.atom_mapping.perses_scorers import default_perses_scorer
from openmm import unit
from collections import OrderedDict

# load ligands
ligands_supp = Chem.SDMolSupplier('../02_ligands/ligands.sdf', removeHs=False)
ligand_mols = [SmallMoleculeComponent(m) for m in ligands_supp]

# create minimum spanning graph
mapper = PersesAtomMapper(allow_ring_breaking=False, preserve_chirality=True,
                          use_positions=True, coordinate_tolerance=0.2*unit.angstrom)

scorer_fn = partial(default_perses_scorer, use_positions=True, normalize=False)

network = generate_minimal_spanning_network(ligand_mols, [mapper,], default_perses_scorer)

# Get edges
edges = [edge for edge in network.edges]

# Analyze distance overlap for manual validation
for i, edge in enumerate(edges):
    if np.any(edge.get_distances() > 1):
        name = f"{edge.componentA.name}_{edge.componentB.name}"
        print(f"{name} exceeds 1 A pairwise distance")

output = {}
output['remarks'] = "Minimum spanning graph using Perses mapper and scorer"
output['planner'] = "openfe v0.9.2 minimal_spanning_graph"

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
            'method': "Perses default geometric scorer",},
        'mapper': "Perses 0.10.1 (allow_ring_breaking=False, preserve_chirality=True, use_positions=True, coordinate_tolerance=0.2)",
        'remarks': None,
        }

output['edges'] = edges_dict

with open('01_perses_mst_geom_score_openfe.yml', 'w') as yaml_file:
    yaml.dump(output, yaml_file, default_flow_style=None, sort_keys=False)
