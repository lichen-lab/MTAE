#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:50:36 2022

@author: chen61
"""


from __future__ import print_function
import os
import sys
import argparse

import warnings
warnings.filterwarnings('ignore')

import time



from libs_hp import *




def parse_arguments(parser):
    
    parser.add_argument('--hdf5', type=str,
                        help='hdf5 file')
    
    
    parser.add_argument('--comparetype', type=str,
                        help='cn or mci')
    
    parser.add_argument('--isnorm', type=str,
                        help='no or yes')
    
    parser.add_argument('--AEtype', type=str,
                        help='define or custom')  
    
    
    parser.add_argument('--nrep', type=int,
                        help='number of experiments')  
    
       
    parser.add_argument('--methods', nargs='+' ,type=str, default=[
                           'LSTMAE','CNNAE'],
                        help='Methods used in comparison')
    
    
    args = parser.parse_args()

    return args
    
    
    
    
def main(args):

    print(args)
    
    
    hdf5=args.hdf5
    comparetype=args.comparetype
    isnorm=args.isnorm
    AEtype=args.AEtype
    nrep=args.nrep
    methods=args.methods
    
                  

    auc_all,acc_all,f1_all,mcc_all,auprc_all=benchmark(hdf5=hdf5,
                  methods=methods,
                  nrep=nrep,test_split=0.2,comparetype=comparetype,
                  isnorm=isnorm,AEtype=AEtype)


    auc_all_df = pd.DataFrame(data = auc_all, columns = methods)
    acc_all_df = pd.DataFrame(data = acc_all, columns = methods)
    f1_all_df = pd.DataFrame(data = f1_all, columns = methods)
    mcc_all_df = pd.DataFrame(data = mcc_all, columns = methods)
    auprc_all_df = pd.DataFrame(data = auprc_all, columns = methods)


    outh5=comparetype+'.'+isnorm+'.h5'

    hf = h5py.File(outh5, 'w')
    hf.create_dataset('auc_'+isnorm, data=auc_all_df)
    hf.create_dataset('acc_'+isnorm, data=acc_all_df)
    hf.create_dataset('f1_'+isnorm, data=f1_all_df)
    hf.create_dataset('mcc_'+isnorm, data=mcc_all_df)
    hf.create_dataset('auprc_'+isnorm, data=auprc_all_df)
    hf.close()


    outputResult(auc_all,acc_all,f1_all,mcc_all,auprc_all,methods,comparetype,isnorm)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training model')
    args = parse_arguments(parser)
    print(args)
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
    




