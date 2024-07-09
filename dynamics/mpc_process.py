import pandas as pd
import multiprocessing
from tqdm import tqdm

chunksize = 10 ** 6  # chunksize for reading in pandas
obs_codes = ['G96', '703']  # observatory codes for CSS
n_lines = 425059248  # counted using sed -n '$=' NumObs.txt

# Define a function to be applied to each chunk
def process_chunk(chunk):
    chunk.columns = ['packed_desig', 'year', 'obs']
    # get all rows where the last three digits are either G96 or 703 (the CSS observatories) and between 2005 and 2012
    processed_chunk = chunk[((chunk.obs.astype('string').str[-3:] == 'G96') | (chunk.obs.astype('string').str[-3:] == '703')) &
    ((chunk.year.astype('string').str[-4:].astype('int') > 2005) & (chunk.year.astype('string').str[-4:].astype('int') < 2012))]
    # write to csv
    processed_chunk.to_csv('mpc_data/mpc_processed_' + str(i) + '.csv')
    return None

# Read the large DataFrame in chunks
chunks = pd.read_fwf('NumObs.txt', sep=None, chunksize=chunksize, header=None, usecols=[0, 1, 10], quoting=3, on_bad_lines='skip')

# # pool
# pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)  # avoids locking up the CPU by using every core
# result = tqdm(pool.imap(process_chunk, zip(enumerate(chunks))))

for i, chunk in tqdm(enumerate(chunks), total=int(n_lines/chunksize)):  # approximate total number of iterations
    process_chunk(chunk)  # process chunk