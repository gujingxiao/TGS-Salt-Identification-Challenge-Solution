import numpy as np
import pandas as pd
from tqdm import tqdm

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)

"""
used for converting the decoded image to rle mask
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

df_1 = pd.read_csv('../submission/submission_1.csv')
df_2 = pd.read_csv('../submission/submission_2.csv')
df_3 = pd.read_csv('../submission/submission_3.csv')
df_4 = pd.read_csv('../submission/submission_4.csv')
df_5 = pd.read_csv('../submission/submission_5.csv')

df_1_id = pd.read_csv('../submission/submission_1.csv', index_col="rle_mask")
df_2_id = pd.read_csv('../submission/submission_2.csv', index_col="rle_mask")
df_3_id = pd.read_csv('../submission/submission_3.csv', index_col="rle_mask")
df_4_id = pd.read_csv('../submission/submission_4.csv', index_col="rle_mask")
df_5_id = pd.read_csv('../submission/submission_5.csv', index_col="rle_mask")

d1 = df_1_id.values.tolist()
d2 = df_2_id.values.tolist()
d3 = df_3_id.values.tolist()
d4 = df_4_id.values.tolist()
d5 = df_5_id.values.tolist()


"""
Applying vote on the predicted mask
"""
for i in tqdm(range(df_1.shape[0])):
    name = df_1_id.values[i]
    index_1 = d1.index(name)
    index_2 = d2.index(name)
    index_3 = d3.index(name)
    index_4 = d4.index(name)
    index_5 = d5.index(name)

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

    if str(df_5.loc[index_5, 'rle_mask']) != str(np.nan):
        decoded_mask_5 = rle_decode(df_5.loc[index_5, 'rle_mask'])
    else:
        decoded_mask_5 = np.zeros((101, 101), np.int8)

    decoded_masks = decoded_mask_1 + decoded_mask_2 + decoded_mask_3 + decoded_mask_4 + decoded_mask_5
    decoded_masks = decoded_masks / 5.0

    df_1.loc[index_1, 'rle_mask'] = rle_encode(np.round(decoded_masks >= 0.5))

df_1.to_csv('../submission/vote_correction.csv',index=False)
