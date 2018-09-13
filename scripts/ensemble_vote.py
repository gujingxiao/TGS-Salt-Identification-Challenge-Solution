import numpy as np
import pandas as pd
from tqdm import tqdm
from rle_code import rle_encode, rle_decode

df_1 = pd.read_csv('../submission/submission_1.csv')
df_2 = pd.read_csv('../submission/submission_2.csv')
df_3 = pd.read_csv('../submission/submission_3.csv')
df_4 = pd.read_csv('../submission/submission_4.csv')

df_1_id = pd.read_csv('../submission/submission_1.csv', index_col="rle_mask")
df_2_id = pd.read_csv('../submission/submission_2.csv', index_col="rle_mask")
df_3_id = pd.read_csv('../submission/submission_3.csv', index_col="rle_mask")
df_4_id = pd.read_csv('../submission/submission_4.csv', index_col="rle_mask")

"""
Applying vote on the predicted mask 
"""
for i in tqdm(range(df_1.shape[0])):
    name = df_1_id.values[i]
    index_1 = df_1_id.values.tolist().index(name)
    index_2 = df_2_id.values.tolist().index(name)
    index_3 = df_3_id.values.tolist().index(name)
    index_4 = df_4_id.values.tolist().index(name)

    if str(df_1.loc[index_1,'rle_mask'])!=str(np.nan):
        decoded_mask_1 = rle_decode(df_1.loc[index_1,'rle_mask'])
    else:
        decoded_mask_1 = np.zeros((101, 101), np.int8)

    if str(df_2.loc[index_2, 'rle_mask']) != str(np.nan):
        decoded_mask_2 = rle_decode(df_2.loc[index_2, 'rle_mask'])
    else:
        decoded_mask_2 = np.zeros((101, 101), np.int8)

    if str(df_3.loc[index_3, 'rle_mask']) != str(np.nan):
        decoded_mask_3 = rle_decode(df_3.loc[index_3, 'rle_mask'])
    else:
        decoded_mask_3 = np.zeros((101, 101), np.int8)

    if str(df_4.loc[index_4, 'rle_mask']) != str(np.nan):
        decoded_mask_4 = rle_decode(df_4.loc[index_4, 'rle_mask'])
    else:
        decoded_mask_4 = np.zeros((101, 101), np.int8)

    decoded_masks = decoded_mask_1 + decoded_mask_2 + decoded_mask_3 + decoded_mask_4
    decoded_masks = decoded_masks / 4.0

    df_1.loc[index_1, 'rle_mask'] = rle_encode(np.round(decoded_masks >= 0.5))

df_1.to_csv('../submission/vote_correction.csv',index=False)

