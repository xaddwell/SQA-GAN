# -*- coding: utf-8 -*-
"""
yimingxiao
"""
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error
def eval():
    ptsn_pred_file="NISQA_results.csv"
    ptsn_file="pstn_train.csv"
    tencent_with_file="tencent with.csv"
    tencent_without_file="tencent without.csv"
    withoutReverberationTrainDevMOS_file="withoutReverberationTrainDevMOS.csv"
    withReverberationTrainDevMOS_file="withReverberationTrainDevMOS.csv"
    df1_1=pd.read_csv(tencent_with_file,error_bad_lines=False).iloc[:,[0,1]].sort_values(by='deg')
    df1_2=pd.read_csv(withReverberationTrainDevMOS_file,error_bad_lines=False).iloc[:,[0,1]].sort_values(by='deg_wav')
    df2_1=pd.read_csv(tencent_without_file,error_bad_lines=False).iloc[:,[0,1]].sort_values(by='deg')
    df2_2=pd.read_csv(withoutReverberationTrainDevMOS_file,error_bad_lines=False).iloc[:,[0,1]].sort_values(by='deg_wav')
    df3_1 = pd.read_csv(ptsn_pred_file, error_bad_lines=False).iloc[:, [0, 1]].sort_values(by='deg')
    df3_2 = pd.read_csv(ptsn_file,error_bad_lines=False).iloc[:, [0, 1]].sort_values(by='filename')

    with_mos_pred=df1_1['mos_pred']
    with_mos = df1_2['mos']
    without_mos_pred = df2_1['mos_pred']
    without_mos = df2_2['mos']
    ptsn_mos=df3_2['MOS']
    ptsn_mos_pred=df3_1['mos_pred']

    pccs1 = np.corrcoef(with_mos, with_mos_pred)[0][1]
    pccs2 = np.corrcoef(without_mos, without_mos_pred)[0][1]
    rmse1  = np.sqrt(mean_squared_error(with_mos, with_mos_pred))
    rmse2  = np.sqrt(mean_squared_error(without_mos, without_mos_pred))
    SROCC1 = stats.spearmanr(with_mos, with_mos_pred)[0]
    SROCC2 = stats.spearmanr(without_mos, without_mos_pred)[0]


    pccs3 = np.corrcoef(ptsn_mos, ptsn_mos_pred)[0][1]

    rmse3 = np.sqrt(mean_squared_error(ptsn_mos, ptsn_mos_pred))

    SROCC3 = stats.spearmanr(ptsn_mos, ptsn_mos_pred)[0]
    print("withR_pccs:",pccs1,"withoutR_pccs:",pccs2,"ptsn_pccs:",pccs3)
    print("withR_srocc:",SROCC1,"withoutR_srocc:",SROCC2,"ptsn_srocc:",SROCC3)
    print("withR_rmse:",rmse1,"withoutR_rmse:",rmse2,"ptsn_rmse:",rmse3)

if __name__=='__main__':
    eval()