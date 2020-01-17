"""
ligands.py
Protein-Ligand Benchmark Dataset for testing Parameters and Methods of Free Energy Calculations.
Handles the ligand data
"""

from PLBenchmarks import utils, targets

import re
import pandas as pd
from simtk import unit
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools

from PIL import Image

import io

import yaml
try:
    from importlib.resources import open_text
except ImportError:
    # Python 2.x backport
    from importlib_resources import open_text

class ligand:
    """
    Store and convert the data of one ligand in a 'pandas.Series'.
    """
    
    _observables = ['dg', 'dh', 'tds', 'ki', 'ic50', 'pic50']
   
    def __init__(self, d: dict):
        """
        Store and convert the data of one ligand in a `pandas.Series`.
        :param d: dict with the ligand data
        :return None
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
                
    def deriveObservables(self, derivedObs='dg', dest='DerivedMeasurement', outUnit=unit.kilocalories_per_mole):
        """
        Derive observables from (stored) primary data, which is then stored in the 'pandas.DataFrame'
        :param derivedObs: type of derived observable, can be any of 'dg' 'ki', 'ic50' or 'pic50'
        :param dest: string with column name for 'pandas.DataFrame' where the derived observable should be stored.
        :param outUnit: 'simtk.unit' unit of derived coordinate
        :return: None
        """
        assert derivedObs in self._observables, 'Observable to be derived not known. Should be any of dg, ki, ic50, or pic50'
        for obs in self._observables:
            if ('measurement', obs) in list(self.data.index):
                self.data = self.data.append(pd.Series([utils.convertValue(self.data[('measurement', obs)], obs, derivedObs, outUnit=outUnit), 
                                                        utils.convertValue(self.data[('measurement', f'e_{obs}')], obs, derivedObs, outUnit=outUnit)], 
                                                       index=pd.MultiIndex.from_tuples([(dest, derivedObs), (dest, f'e_{derivedObs}')])
                                                      )
                                            )
                break
        else:
            print(f'Conversion to observable {derivedObs} not possible.')
    
    def getName(self):
        """
        Access the name of the ligand.
        :return: name: string
        """
        return self.data['name'][0]

    def getDF(self, cols=None):
        """
        Access the ligand data as a `pandas.DataFrame`
        :param cols: list of columns which should be returned in the `pandas.DataFrame`
        :return: data: `pandas.DataFrame`
        """
        if cols:
            return self.data[cols]
        else:
            return self.data
    
    def findLinks(self):
        """
        Processes primary data to have links in the html string of the ligand data
        :return: None
        """
        if ('measurement', 'doi') in list(self.data.index):
            doi = self.data['measurement', 'doi']
            if str(doi) != 'nan':
                res = []
                for ddoi in re.split(r'[; ]+', str(doi)):
                    res.append(utils.findDoiUrl(ddoi))
            self.data['measurement', 'doi_html'] = (r'\n').join(res)
        if ('pdb') in list(self.data.index):
            pdb = self.data['pdb']            
            self.data['pdb_html'] = utils.findPdbUrl(pdb)
            
    def getMol(self):
        """
        Get file path relative to the PLBenchmarks repository of the SDF coordinate file of the docked ligand
        :return: fname: string with file path
        """
        fname = self.data['docked'][0]
        print(fname)
        return fname

    def addMolToFrame(self):
        """
        Adds a image file of the ligand to the `pandas.DataFrame`
        :return: None
        """
        PandasTools.AddMoleculeColumnToFrame(self.data, smilesCol='smiles', molCol='ROMol', includeFingerprints=False)

    def getHTML(self, columns=None):
        """
        Access the ligand as a HTML string
        :param columns: list of columns which should be returned in the `pandas.DataFrame`
        :return: html: HTML string
        """
        self.findLinks()
        if columns:
            html = pd.DataFrame(self.data[columns]).to_html()
        else:

            html = pd.DataFrame(self.data).to_html()
        html = html.replace('REP1', '<a target="_blank" href="')
        html = html.replace('REP2', '">')
        html = html.replace('REP3', '</a>')
        html = html.replace("\\n","<br>")
        return html

    def getImg(self):
        """
        Creates a molecule image.
        :return: img: a PIL.Image object
        """
        dr = Draw.MolDraw2DCairo(200,200)
        opts = dr.drawOptions()
        opts.clearBackground=True
        mol = Chem.MolFromSmiles(self.data['smiles'][0])
        Chem.rdDepictor.Compute2DCoords(mol)
        dr.DrawMolecule(mol, legend=self.data['name'][0])
        img = Image.open(io.BytesIO(dr.GetDrawingText())).convert('RGBA')
        datas = img.getdata()
        
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        return img


class ligandSet(dict):
    """
        Class inherited from dict to store all available ligands of one target.
    """
    
    def __init__(self, target, *arg,**kw):
        """
        Initializes ligandSet class
        :param target: string name of target
        :param arg: arguments for dict (base class)
        :param kw: keywords for dict (base class)
        """
        super(ligandSet, self).__init__(*arg, **kw)
        tp = targets.getTargetDataPath(target)      
        file = open_text('.'.join(tp), 'ligands.yml')
        data = yaml.full_load_all(file)    
        for d in data:
            l = ligand(d)
            l.deriveObservables(derivedObs='dg')
            l.findLinks()
            l.addMolToFrame()
            self[l.getName()] = l
        file.close()
          
    def getLigand(self, name):
        """
        Accesses one ligand of the ligandSet
        :param name: string name of the ligand
        :return: ligand: ligand class
        """
        for key in self.keys():
            if key == name:
                return self[key]
                break
        else:
            raise ValueError(f'Ligand {name} is not part of set.')

    def getDF(self, columns=None):
        """
        Access the ligandSet as a 'pandas.DataFrame'
        :param columns: list of columns which should be returned in the `pandas.DataFrame`
        :return: df: `pandas.DataFrame`
        """
        dfs=[]
        for key, item in self.items():
            dfs.append(item.getDF(columns))
        df = pd.DataFrame(dfs)
        return df

    def getHTML(self, columns=None):
        """
        Access the ligandSet as a HTML string
        :param cols: list of columns which should be returned in the `pandas.DataFrame`
        :return: html: HTML string
        """
        df = self.getDF(columns)
        html = df.to_html()
        return html

