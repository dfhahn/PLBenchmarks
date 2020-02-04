"""
edges.py
Functions and classes for handling the perturbation edges.
"""

from PLBenchmarks import utils, targets, ligands

import pandas as pd
import numpy as np
import yaml
try:
    from importlib.resources import open_text
except ImportError:
    # Python 2.x backport
    from importlib_resources import open_text



class edge:
    """
    Store and convert the data of one perturbation ("edge") in a :py:class:`pandas.Series`.
    
    :param d: :py:class:`dict` with the edge data
    :return: None
    """
    
    def __init__(self, d: dict):
        """
        Initialize edge class from a dictionary
        
        :param d: :py:class:`dict` with the edge data
        :return: None
        """
        self.data = pd.Series(d)
        self._name = None


    def addLigData(self, ligs):
        """
        Adds data from ligands to :py:class:`~PLBenchmarks:edges.edge`. Molecule images and the affinity difference are added.
        
        :param ligs: :py:class:`PLBenchmarks:ligands:ligandSet` class of the same target
        :return: None
        """
        l0 = None
        l1 = None
        dg0 = 0
        dg1 = 0
        err0 = 0
        err1 = 0
        for key, item in ligs.items():
            if key == 'lig_' + str(self.data[0]):
                l0 = item.data['ROMol'][0][0]
                dg0 = item.data[('DerivedMeasurement', 'dg')]
                err0 = item.data[('DerivedMeasurement', 'e_dg')]
            if key == 'lig_' + str(self.data[1]):
                l1 = item.data['ROMol'][0][0]
                dg1 = item.data[('DerivedMeasurement', 'dg')]
                err1 = item.data[('DerivedMeasurement', 'e_dg')]
        self.data['Mol1'] = l0
        self.data['Mol2'] = l1
        self.data['exp. DeltaG [kcal/mol]'] = round(dg1-dg0, 2)
        self.data['exp. Error [kcal/mol]'] = round(np.sqrt(np.power(err0, 2.0)+ np.power(err1, 2.0)), 2)

    def getDF(self, cols=None):
        """
        Access the edge data as a :py:class:`pandas.DataFrame`

        :param cols: list of columns which should be returned in the :py:class:`pandas.DataFrame`
        :return: :py:class:`pandas.DataFrame`
        """
        if cols:
            return self.data[cols]
        else:
            return self.data

    def getDict(self):
        """
        Access the edge data as a :py:class:`dict` which contains the name of the edge as key and the names of the two ligands as :py:class:`list`.
        
        :return: :py:class:`dict`
        """
        return {f'edge_{self.data[0]}_{self.data[1]}': [f'lig_{self.data[0]}', f'lig_{self.data[1]}']}

    def getName(self):
        """
        Access the name of the edge.
        
        :return: name as string
        """
        if self._name is not None:
            return self._name
        else:
            return f'edge_{self.data[0]}_{self.data[1]}'
    

class edgeSet(dict):
    """
    Class inherited from dict to store all available edges of one target.
    """
    
    def __init__(self, target, *arg,**kw):
        """
        Initializes edgeSet class
        
        :param target: string name of target
        :param arg: arguments for :py:class:`dict` (base class)
        :param kw: keywords for :py:class:`dict` (base class)
        """
        super(edgeSet, self).__init__(*arg, **kw)
        tp = targets.getTargetDataPath(target)
        ligs = ligands.ligandSet(target)
        file = open_text('.'.join(tp), 'edges.yml')
        data = yaml.full_load_all(file)
        for d in data:
            e = edge(d)
            e.addLigData(ligs)
            self[e.getName()] = e

    def getEdge(self, name):
        """
        Accesses one edge of the :py:class:`PLBenchmarks.edges.edgeSet`
        
        :param name: string name of the edge
        :return: :py:class:`PLBenchmarks:edges:edge` class
        """
        for key in self.keys():
            if key == name:
                return self[key]
                break
        else:
            raise ValueError(f'Edge {name} not part of set.')

    def getDF(self, columns=None):
        """
        Access the :py:class:`PLBenchmarks:edges.edgeSet` as a :py:class:`pandas.DataFrame`
        
        :param cols: :py:class:`list` of columns which should be returned in the :py:class:`pandas.DataFrame`
        :return: :py:class:`pandas.DataFrame`
        """
        dfs=[]
        for key, item in self.items():
            dfs.append(item.getDF(columns))
        df = pd.DataFrame(dfs)
        return df

    def getHTML(self, columns=None):
        """
        Access the :py:class:`PLBenchmarks:edges.edgeSet` as a HTML string
        
        :param cols: :py:class:`list` of columns which should be returned in the :py:class:`pandas.DataFrame`
        :return: HTML string
        """
        df = self.getDF(columns)
        html = df.to_html()
        return html

    def getDict(self):
        """
        Access the :py:class:`PLBenchmarks:edges.edgeSet` as a dict which contains the name of the edges as key and the names of the two ligands in a list as items.

        :return: :py:class:`dict`
        """
        res = {}
        for key, item in self.items():
            res.update(item.getDict())
        return res
        
