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
    ['notation', re.compile('[a-zA-Z]+(?=-RNN)'),str],
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
compileParameter=partial(compileForPatterns, patterns=argsParameters)
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

    trainDFPerDataName={}

    def __init__(self, experiment_name, experiment_dir="experiments/", loadCSV=True, novel=False):

        self._PATH=os.path.join(experiment_dir, experiment_name)
        *_, dataName , paramName = os.path.split(experiment_name)

        self.files = files = list(os.listdir(self._PATH))

        if not novel:
            self.csvNames = [file for file in files if '.csv' in file and not 'metrics' in file and 'novels' not in file ]
        else:
            self.csvNames = [file for file in files if '.csv' in file and not 'metrics' in file and 'novels' in file ]

        self.logNames = [file for file in files if '.txt' in file and 'log' in file ]
        self.metricNames = [file for file in files if '.csv' in file and 'metrics' in file and 'novels' in file ]

        self.dataInfo= dataInfo =compileDataset(dataName)
        self.params= params =compileParameter(paramName)

        if dataInfo['notation'].lower() == 'fragsmiles':
            class_=SampledMoleculesFragSm
        else:
            class_ = SampledMoleculesSm

        self.gens =pd.DataFrame()

        for csvName in self.csvNames:
            if not loadCSV:
                break
            setup=compileGen(csvName)
            dataFile = f'{dataInfo["dataset"]}_train_{setup["fold"]}.tar.xz' if setup["fold"]!=None else f'{dataInfo["dataset"]}_train.tar.xz'

            if not dataFile in self.trainDFPerDataName:
                trainDF=pd.read_csv(os.path.join(experiment_dir,'data',dataFile), usecols = ["smiles"], compression="xz", )
                if novel:
                    trainDF['scaff']=trainDF['smiles'].apply(lambda x: MurckoScaffold.MurckoScaffoldSmiles(smiles=x, includeChirality=True) )
                self.trainDFPerDataName[dataFile]=trainDF

            df = pd.read_csv(os.path.join(self._PATH, csvName))
            sampled = class_(df, self.trainDFPerDataName[dataFile])

            self.gens=pd.concat([
            self.gens, pd.DataFrame([ {**dataInfo ,**params, **setup, 'sampled':sampled} ])
            ], ignore_index=True)

        self.logs=pd.DataFrame()
        for logName in self.logNames:
            fold=compileLoss(logName)['fold']
            fold = int(fold) if fold else -1

            log=adjustDFloss( pd.read_csv(os.path.join(self._PATH, logName)).assign(fold=fold) )
            self.logs=pd.concat([ self.logs, log ], ignore_index=False)

    def getTrainTestLosses(self):
        train = self.logs.query('mode=="Train" and fold!=""').groupby('epoch', as_index=True)['running_loss'].describe().loc[:,['mean','std']]
        test = self.logs.query('mode=="Eval" and fold!=""').groupby('epoch', as_index=True)['running_loss'].describe().loc[:,['mean','std']]
        return train, test
    
    def getResultsGens(self):
        results = self.gens.copy()

        results[['valid','unique','novel']]=results['sampled'].apply(lambda x: x.getMetricsAsDF())

        results.drop(columns='sampled', inplace=True)
        return results.sort_values(['amount','fold','epoch'])

    def getChiralResultsGens(self):
        results = self.gens.copy()

        results[['chirals','invalid','valid','unique','novel']]=results['sampled'].apply(lambda x: x.getChiralMetricsAsDF())

        results.drop(columns='sampled', inplace=True)
        return results.sort_values(['amount','fold','epoch'])

    def getResultsNovels(self):
        metrics=pd.DataFrame()
        for novel in self.metricNames:
            setup=compileGen(novel)
            df = pd.read_csv(os.path.join(self._PATH, novel), index_col=0, header=None, names=[0]).T
            df = pd.concat([ pd.DataFrame([ {**self.dataInfo ,**self.params, **setup} ]), df ], axis=1)

            metrics = pd.concat([ metrics, df,], ignore_index=True)
        
        dropCols = [column for column in metrics.columns if any( m in column.lower() for m in ['val','uniq','novel','testsf'] ) ]

        return metrics.drop(columns=dropCols)

    def getScaffoldResults(self):
        results = self.gens.copy()

        results[['total','unique','novel','chiral']]=results['sampled'].apply(lambda x: x.getScafMetricsAsDF())

        results.drop(columns='sampled', inplace=True)
        return results.sort_values(['amount','fold','epoch'])

    def __repr__(self) -> str:
        return self._PATH
        