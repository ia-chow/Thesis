import pandas as pd
from tqdm import tqdm

chunksize = 10 ** 6  # chunksize for reading in pandas
obs_codes = ['G96', '703']  # observatory codes for CSS
n_lines = 425059248  # counted using sed -n '$=' NumObs.txt

# Define a function to be applied to each chunk
def process_chunk(chunk):
    # print(chunk)
    chunk.columns = ['packed_desig', 'year', 'obs']
    # get all rows where the last three digits are either G96 or 703 (the CSS observatories) and between 2005 and 2012 inclusive
    processed_chunk = chunk[((chunk.obs.astype('string').str[-3:] == 'G96') | (chunk.obs.astype('string').str[-3:] == '703')) &
    ((chunk.year.astype('string').str[-4:].astype('int') >= 2005) & (chunk.year.astype('string').str[-4:].astype('int') <= 2012))]
    # write to csv
    processed_chunk.to_csv('mpc_data_unn/mpc_processed_' + str(i) + '.csv')
    print(processed_chunk.packed_desig.value_counts())
    return None

# Read the large DataFrame in chunks
chunks = pd.read_fwf('UnnObs.txt', chunksize=chunksize, sep=None, header=None, usecols=[0, 1, 11], quoting=3, on_bad_lines='skip')

for i, chunk in tqdm(enumerate(chunks)):  # approximate total number of iterations
    process_chunk(chunk)  # process chunk