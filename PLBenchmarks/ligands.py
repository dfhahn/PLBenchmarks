"""
ligands.py
Protein-Ligand Benchmark Dataset for testing Parameters and Methods of Free Energy Calculations.
Handles the ligand data
"""

from .util import *

import re
import yaml
import pandas as pd
from simtk import unit
from rdkit.Chem import PandasTools

class ligand:
    
    """
        Store and convert the data of one ligand in a pandas Series.
    """
    
    _observables = ['dg', 'dh', 'tds', 'ki', 'ic50', 'pic50']
   
    def __init__(self, d: dict):
        """
            Initialize ligand class from dictionary
        """
        self.data = pd.Series(d)
        # Expand measurement dict into single fields
        if 'measurement' in list(self.data.index):
            meas = pd.Series(self.data['measurement'])
            meas.index = ['measurement:' + c for c in meas.index]
            self.data.drop(['measurement'], inplace=True)
            self.data = pd.concat([self.data, meas])
            index = self.data.index.to_series().str.split(':', expand=True).fillna('')
            self.data.index = pd.MultiIndex.from_arrays([index[0].tolist(), index[1].tolist()])
            for obs in self._observables:
                if ('measurement', obs) in list(self.data.index):
                    self.data = self.data.append(pd.Series([0], 
                                              index=pd.MultiIndex.from_tuples([('measurement', f'e_{obs}')])
                                             )
                                )
                    vals = self.data[('measurement', f'{obs}')]
                    u = unit.dimensionless
                    if vals[2] == 'nM':
                        u = unit.nano*unit.molar
                    elif vals[2] == 'uM':
                        u = unit.micro*unit.molar
                    elif vals[2] == 'kj/mol':
                        u = unit.kilojoules_per_mole
                    elif vals[2] == 'kcal/mol':
                        u = unit.kilocalories_per_mole
                    self.data[('measurement', f'e_{obs}')] = unit.quantity.Quantity(vals[1], u)
                    self.data[('measurement', obs)] = unit.quantity.Quantity(vals[0], u)
                
    def deriveObservables(self, derivedObs='dg', dest='DerivedMeasurement'):
        """
            Calculate 
        """
        assert derivedObs in self._observables, 'Observable to be derived not known. Should be any of dg, ki, ic50, or pic50'
        for obs in self._observables:
            if ('measurement', obs) in list(self.data.index):
                self.data = self.data.append(pd.Series([convertValue(self.data[('measurement', obs)], obs, derivedObs), 
                                                        convertValue(self.data[('measurement', f'e_{obs}')], obs, derivedObs)], 
                                                       index=pd.MultiIndex.from_tuples([(dest, derivedObs), (dest, f'e_{derivedObs}')])
                                                      )
                                            )
                break
        else:
            print(f'Conversion to observable {derivedObs} not possible.')
    
    def getDF(self, cols=None):
        if cols:
            return self.data[cols]
        else:
            return self.data
    
    def findLinks(self):
        if ('measurement', 'doi') in list(self.data.index):
            doi = self.data['measurement', 'doi']
            if str(doi) != 'nan':
                res = []
                for ddoi in re.split(r'[; ]+', str(doi)):
                    res.append(findDoiUrl(ddoi))
            self.data['measurement', 'doi_html'] = (r'\n').join(res)
        if ('pdb') in list(self.data.index):
            doi = self.data['pdb']            
            self.data['pdb_html'] = findPdbUrl(pdb)
    
    def addMolToFrame(self):
        PandasTools.AddMoleculeColumnToFrame(self.data, smilesCol='smiles', molCol='ROMol', includeFingerprints=False)

    def getHTML(self):
        self.findLinks()
        html = pd.DataFrame(self.data).to_html()
        html = html.replace('REP1', '<a target="_blank" href="')
        html = html.replace('REP2', '">')
        html = html.replace('REP3', '</a>')
        html = html.replace("\\n","<br>")
        return html
