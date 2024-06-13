"""
This code documents and shows how the data is gathered from the different
Corpora used within articubench. This code will only run if you have access to
the full corpora and have put them in the right path.

.. warning::

    This module is not in any consistent state! At the moment this is a
    collection of code snippeds on how the data is generated in the first
    place. This is not needed to run the benchmark.

"""

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# TODO clean this file

######################################################
### KEC /ja/ /halt/ (Sering & Tomaschek ESSV 2020) ###
######################################################

# R code
'''
dat <- readRDS("/home/tino/Documents/phd/projects/essv2020/data/essv2020.rds")
start_idx <- which(dat$AR.start.word == TRUE)
#end_idx <- c((start_idx - 1)[-1], nrow(dat))  # subtract one from the start_idx, remove the first one and add the last row index

#words <- dat[start_idx, c('Speaker', 'Word', 'WordStartTime', 'WordEndTime', 'Time', 'SenTT.X', 'SenTT.Y', 'SenTT.Z', 'SenTB.X', 'SenTB.Y', 'SenTB.Z')]
words <- dat[, c('Speaker', 'Word', 'WordStartTime', 'WordEndTime', 'Time', 'SenTT.X', 'SenTT.Y', 'SenTT.Z', 'SenTB.X', 'SenTB.Y', 'SenTB.Z', 'AR.start.word')]

write.csv(words, 'data/KEC/KEC_ja_halt.csv', row.names=FALSE)
'''

# read the data processed in R
essv2020 = pd.read_csv('data/KEC/KEC_ja_halt.csv')

#  wide format
# ------------
# Creating a wide format where every time series is stored as np.array in one
# cell of the pandas Data.Frame.

# TODO: Do we need this? How to connect to phone sequences?
essv2020['rownr'] = list(range(len(essv2020)))

# 1.1 essv2020
start_indices = list(essv2020[essv2020['AR.start.word'] == True]['rownr'])
start_indices.append(len(essv2020))

new_rows = list()
for ii, start_index in enumerate(start_indices[:-1]):
    end_index = start_indices[ii + 1]  # excluding
    row = essv2020.iloc[start_index,]
    speaker = row['Speaker']
    word = row['Word']
    start_time = row['WordStartTime']
    end_time = row['WordEndTime']
    rows = essv2020.iloc[start_index:end_index,]
    time = np.array(rows['Time'])
    senttx = np.array(rows['SenTT.X'])
    sentty = np.array(rows['SenTT.Y'])
    senttz = np.array(rows['SenTT.Z'])
    sentbx = np.array(rows['SenTB.X'])
    sentby = np.array(rows['SenTB.Y'])
    sentbz = np.array(rows['SenTB.Z'])

    new_rows.append(pd.DataFrame({'Speaker': speaker, 'Word': word,
        'WordStartTime': start_time, 'WordEndTime': end_time,
        'Time': [time],
        'SenTT.X': [senttx], 'SenTT.Y': [sentty], 'SenTT.Z': [senttz],
        'SenTB.X': [sentbx], 'SenTB.Y': [sentby], 'SenTB.Z': [sentbz]}))

essv_wide = pd.concat(new_rows, ignore_index=True)

essv_wide.to_pickle('ja_halt_no-audio.pickle')


# copy ja_halt_no-audio.pickle to server


# add signal and sampling_rate on server marser
import pandas as pd

PATH_KEC = "/mnt/shared/corpora/German.KEC/Wav_processed_no_names"

ja_halt = pd.read_pickle('ja_halt_no-audio.pickle')
speaker = None
signals = list()
sample_rates = list()
for ii, row in tqdm(list(ja_halt.iterrows())):
    if speaker != row.Speaker:
        speaker = row.Speaker
        speaker_sig, speaker_sr = sf.read(f"{PATH_KEC}/{speaker}.wav")
    start = int(row.Time[0] * speaker_sr)
    # add 0.005 seconds to the end and remove one sample
    end = int((row.Time[-1] + 0.005) * speaker_sr) - 1
    signals.append(speaker_sig[start:end].copy())
    sample_rates.append(speaker_sr)
ja_halt['Signal'] = signals
ja_halt['SampleRate'] = sample_rates

ja_halt.to_pickle('ja_halt.pickle')


# copy ja_halt.pickle back to local machine



"""
from articubench.util import inv_normalize_cp
import pandas as pd

from articubench.eval_tongue_height import tongue_heights_from_cps

data = pd.read_pickle('articubench/data/tiny_prot4.pkl')
tmp = pd.read_pickle('articubench/data/geco_tiny.pkl')

data['reference_cp'] = None
data['reference_cp'].iloc[:2] = tmp.cp_norm.apply(inv_normalize_cp)

data['reference_tongue_height'] = None
data['reference_ema'] = None

data.to_pickle('articubench/data/tiny.pkl')
"""

