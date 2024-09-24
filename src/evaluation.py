import pandas as pd
import numpy as np
import torch
import os
import re
from functools import partial
from rdkit.Chem.Scaffolds import MurckoScaffold



## Given from stackoverflow :)
def round_sig_figs(val, val_err, sig_figs=2):
    '''
    Round a value and its error estimate to a certain number 
    of significant figures (on the error estimate).  By default 2
    significant figures are used.
    '''
    if val_err == 0.0:
        return val, val_err
    n = int(np.log10(val_err))  # displacement from ones place
    if val_err >= 1:
        n += 1

    scale = 10 ** (sig_figs - n)
    val = round(val * scale) / scale
    val_err = round(val_err * scale) / scale

    return val, val_err

def ConcatMeanStd(row):
    mean, std = row
    mean, std = round_sig_figs(mean, std, 1)

    strng = f'{mean} ± {std}'
    return strng

def ResultsFromMeanStdDF(df):
    names = df.columns.get_level_values(0)
    newdf = pd.DataFrame(index=df.index)
    for name in names:
        newdf[name]=df[name].apply(ConcatMeanStd, axis=1)
    
    return newdf

def cleanTokens(arr):
    toDel=["<bos>","<eos>","<pad>"]
    arr=np.array(arr)
    ids=np.where(  np.logical_or.reduce([arr==i for i in toDel]) )
    return np.delete(arr, ids, axis=None)

def adjustDFloss(df):
    grps=df.groupby('mode')
    train = grps.get_group('Train').reset_index(drop=True)
    eval = grps.get_group('Eval').reset_index(drop=True)

    conc = pd.concat([train,eval], ignore_index=False)
    conc.index.name = 'epoch'

    return conc

compilerPatterns=pd.DataFrame(
    [
    ['dataset', re.compile('^[a-z]+(?=_|Aug)'),str],
    ['aug', re.compile('(?<=Aug)\d+'),int],
    ['notation', re.compile('[a-zA-Z\-]+(?=-RNN)'),str],
    ['hl', re.compile('\d+(?=hl)'),int],
    ['hu', re.compile('\d+(?=hu)'),int],
    ['es', re.compile('\d+(?=es)'),int],
    ['lr', re.compile('[\d\.]+(?=lr)'),float],
    ['bs', re.compile('\d+(?=bs)'),int],
    ['fold',re.compile('(?<=rnn_)\d+(?=_)'),int],
    ['amount',re.compile('(?<=generated)\d+'),int],
    ['epoch',re.compile('(?<=_)\d+(?=_T|_seed)'),int],
    ['T',re.compile('(?<=T)[\d\.]+(?=\.csv|_)'),float],
    # ['seed', re.compile('(?<=seed)\d+(?=_)'),int],
    ], columns=['arg','pattern','type']).set_index('arg')


argsDataset=['dataset','aug','notation']
argsParameters=['hl','hu','bs','es','lr']
argsGen=['fold','amount','epoch',
        #  'seed',
         'T']
argsLoss=['fold']

def compileForPatterns(strng, patterns):
    # return {k: None if compilerPatterns.loc[k,'pattern'].search(strng) == None  \
    #         else compilerPatterns.loc[k,'type'](compilerPatterns.loc[k,'pattern'].search(strng).group())
    #          for k in patterns  }
    dict_={}
    for key in patterns:
        value = compilerPatterns.loc[key,'pattern'].search(strng)
        if value:
            fnc = compilerPatterns.loc[key,'type']
            value = fnc ( value.group() )
        dict_[key]=value
    return dict_

compileDataset = partial(compileForPatterns, patterns=argsDataset)
compileSetup=partial(compileForPatterns, patterns=argsParameters)
compileLoss=partial(compileForPatterns, patterns=argsLoss)
compileGen=partial(compileForPatterns, patterns=argsGen)

chiralCompiler = re.compile(r'(?<!@)@{1,2}(?!@)')

class SampledMolecules:
    def __init__(self, df, trainSm=None):

        self.trainSm = trainSm

        self.scaffolds=None
        self.sampled = df
        self._sm = sm = df["smiles"]

        self._maskValid = ~sm.isna()
        self._maskUnique = (self._maskValid & ~sm.duplicated(keep=False))
        self._maskNovel = None

        if trainSm is not None:
            self._maskNovel = (self._maskUnique & ~sm.isin(trainSm['smiles']))

    def smiles(self, mask = None):
        if mask:
            maskAttr = getattr(self, "_mask"+mask.capitalize())
            return self._sm[maskAttr]

        return self._sm

    def getMetricsAsDF(self):
        return pd.Series([
            self._maskValid.sum(),
            self._maskUnique.sum(),
            self._maskNovel.sum(),
        ])
    
    def _processScaffolds(self):
        sm = self.smiles('valid')
        scaff = sm.apply(lambda x: MurckoScaffold.MurckoScaffoldSmiles(smiles=x, includeChirality=True) )
        self.scaffolds = scaff = scaff.drop( scaff[scaff==""].index)

        self.uniqScaff=scaff.drop_duplicates()
        self.novelScaff = self.uniqScaff[ ~self.uniqScaff.isin(self.trainSm['scaff'].values) ]
        self.chirScaff = self.novelScaff[ self.novelScaff.str.contains('@',regex=False) ]

    def getScafMetricsAsDF(self):
        if self.scaffolds is None:
            try :
                self._processScaffolds()
            except Exception as e:
                print(self, e)
                return pd.Series([0,0,0])

        total = len(self.scaffolds)
        unique=len(self.uniqScaff)
        novel=len(self.novelScaff)

        # da cancellare da qua poi !!
        self.chirScaff = self.novelScaff[ self.novelScaff.str.contains('@',regex=False) ]
        chiral=len(self.chirScaff)

        # return {"total":total, "unique":unique,"novel":novel}
        return pd.Series([ total,unique,novel, chiral ])

class SampledMoleculesSm(SampledMolecules):
    def __init__(self, df, trainSm=None):
        super().__init__(df, trainSm)

        self.nStereoSampled = self.sampled.iloc[:,1].astype(str).apply( lambda x: len( chiralCompiler.findall(x)) )
        self.nStereoGen = self._sm.astype(str).apply(lambda x: len( chiralCompiler.findall(x) ) )
        self.chiral = self.nStereoSampled>0

    def getChiralMetricsAsDF(self):
        return pd.Series([
            self.chiral.sum(),
            (  (self.chiral & ~self._maskValid) | (self.chiral & self._maskValid & (self.nStereoSampled!=self.nStereoGen))).sum(),
            (  self.chiral & self._maskValid & (self.nStereoSampled==self.nStereoGen)).sum(),
            (  self.chiral & self._maskUnique & (self.nStereoSampled==self.nStereoGen)).sum(),
            (  self.chiral & self._maskNovel & (self.nStereoSampled==self.nStereoGen)).sum(),
        ])

class SampledMoleculesFragSm(SampledMolecules):
    def __init__(self, df, trainSm=None):
        super().__init__(df, trainSm)

        self._fSm = fSm = df.iloc[:,1:].apply(lambda x: cleanTokens( x.dropna() ), axis=1)
        # self._fSm_withBegin = df.apply(lambda x : cleanTokens(x.dropna(), ["<eos>","<pad>"]), axis=1)
        self.chiral = fSm.astype(str).str.contains("\||R>|S>",regex=True)

    def fSmiles(self, mask = None):
        if mask:
            maskAttr = getattr(self, "_mask"+mask.capitalize())
            return self._fSm[maskAttr]

        return self._fSm

    # def fSmilesWithBegin(self, mask = None):
    #     if mask:
    #         maskAttr = getattr(self, "_mask"+mask.capitalize())
    #         return self._fSm_withBegin[maskAttr]

    #     return self._fSm_withG

    def getChiralMetricsAsDF(self):
        return pd.Series([
            self.chiral.sum(),
            (~self._maskValid & self.chiral ).sum(),
            (self._maskValid & self.chiral ).sum(),
            (self._maskUnique & self.chiral ).sum(),
            (self._maskNovel & self.chiral ).sum(),
        ])

class Evaluator():

    # train data of smiles notation. Only smiles beacause we just need it to compare and evaluate metrics
    # first key : {dataset}{aug}
    # second key : {fold}

    train_data={}

    def __init__(self, experiment_name, experiment_dir="experiments/"):

        self.full_path=os.path.join(experiment_dir, experiment_name)

        self.csvSamples = []
        self.txtLogs = []
        self.csvNovels = []
        self.csvNovelsMetrics = []

        for file in os.listdir(self.full_path):
            if 'log.txt' in file:
                self.txtLogs.append(file)
            elif 'metrics.csv' in file and 'novels' in file:
                self.csvNovelsMetrics.append(file)
            elif not 'descriptors.csv' in file and 'novels' in file:
                self.csvNovels.append(file)
            elif not 'novels' in file and not 'descriptors.csv' in file and 'generated' in file:
                self.csvSamples.append(file)

        splitted = os.path.normpath(self.full_path).split(os.sep)
        *_, dataName, setupName = splitted

        self.datasetArgs= compileDataset(dataName)
        self.setupArgs= compileSetup(setupName)

        self.data_name = data_name = self.datasetArgs['dataset'] + ('Aug5' if self.datasetArgs['aug']==5 else '')

        if not data_name in self.train_data:
            data_path = os.path.join('data', data_name)
            self.train_data[data_name] = {}
            data_full = pd.read_csv(data_path + '.tar.xz', usecols=['smiles']+ [f'fold{fold}' for fold in range(5)], compression='xz' )
            for fold in range(5):
                query_str = f' fold{fold} == "train" '
                self.train_data[data_name][fold]=data_full.query(query_str)[['smiles']]

        self.samples = None
        self.logs = None

    def load_logs(self):
        if not self.txtLogs:
            return False
        
        self.logs=pd.DataFrame()
        for log_name in self.txtLogs:
            fold=compileLoss(log_name)['fold']

            log=adjustDFloss( pd.read_csv(os.path.join(self.full_path, log_name)).assign(fold=fold) )
            self.logs=pd.concat([ self.logs, log ], ignore_index=False)

        return True
    
    def load_samples(self):

        if not self.csvSamples:
            return False

        if self.datasetArgs['notation'].lower() == 'fragsmiles':
            class_=SampledMoleculesFragSm
        else:
            class_ = SampledMoleculesSm

        self.samples = pd.DataFrame()

        for samples_name in self.csvSamples:
            params = compileGen(samples_name)

            smiles = pd.read_csv(os.path.join(self.full_path, samples_name))
            sampled = class_(smiles, self.train_data[self.data_name][params['fold']])

            df = pd.DataFrame([ { **params, 'sampled':sampled } ])

            self.samples = pd.concat([self.samples, df], ignore_index=True)
        
        return True
    
    def getResultsGens(self):
        results = self.samples.copy()

        results[['valid','unique','novel']]=results['sampled'].apply(lambda x: x.getMetricsAsDF())
        # results.insert(loc=0, column='dataset', value=self.data_name)
        setup = {k:v for k,v in self.setupArgs.items() if k!='es'}
        results = results.assign(dataset = self.data_name + f'_{self.datasetArgs["notation"]}-RNN', **setup)

        results.drop(columns='sampled', inplace=True)
        return results.sort_values(['amount','fold','epoch'])


    def get_as_DFcell(self):
        setup = {k:v for k,v in self.setupArgs.items() if k!='es'}
        df = pd.DataFrame( [{**setup, self.data_name + f'_{self.datasetArgs["notation"]}-RNN' : self}] )
        # df.set_index(list(setup.keys()), inplace=True)
        return df
    
    def getTrainTestLosses(self):
        train = self.logs.query('mode=="Train" and fold!=""').groupby('epoch', as_index=True)['running_loss'].describe().loc[:,['mean','std']]
        test = self.logs.query('mode=="Eval" and fold!=""').groupby('epoch', as_index=True)['running_loss'].describe().loc[:,['mean','std']]
        return train, test
    
    def plot_logs(self, ax, **kwargs):
        stdKwargs = dict(alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                        linewidth=4, antialiased=True)
        
        stdKwargs.update(kwargs)
        train, test = self.getTrainTestLosses()

        for lossDF, label in zip( [train,test], ['train','valid'] ):
            ax.plot(lossDF['mean'], label = label )
            ax.fill_between(lossDF.index, lossDF['mean']-lossDF['std'], 
                            lossDF['mean']+lossDF['std'], **stdKwargs)
            
        setup = [f'{k}={v}' for k,v in self.setupArgs.items() if k!='es']

        ax.set_title(' '.join(setup))