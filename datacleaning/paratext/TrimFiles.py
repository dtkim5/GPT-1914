# TrimFiles.py

# This script uses an already-trained page prediction model to generate predictions
# for pages of text files, which it expects to be already represented as rows
# is a TSV file. The script then uses the predictions to identify start and
# end pages for each file, and actually trims the files, outputting them to a
# new folder.
#
# At the same time it removes running headers from the pages of the files.

import os
import sklearn
import argparse
import numpy as np
import pandas as pd
import header
import joblib

with open('romannumerals.txt', encoding = 'utf-8') as file:
    romannumerals = [x.strip() for x in file.readlines()]

# Load the saved model
clf = joblib.load('models/RF_model4.pkl')

# Command-line arguments
# -t --tsv: the input TSV file, containing page-level predictions
# -i --input: the input directory containing the text files
# -o --output: the output directory to write the trimmed files to

parser = argparse.ArgumentParser(description='Trim text files based on page predictions')
parser.add_argument('-t', '--tsv', help='The input TSV file', required=True)
parser.add_argument('-i', '--input', help='The input directory', required=True)
parser.add_argument('-o', '--output', help='The output directory', required=True)

args = parser.parse_args()
input_file = args.tsv
input_dir = args.input
output_dir = args.output

# Read the TSV file, which will have been created by the script ApplyParatextModel.py
pages = pd.read_csv(input_file, sep='\t')
htids = pages['htid']
pages = pages.drop('htid', axis=1)
probabilities = clf.predict_proba(pages)

# Create a dataframe with columns for pagenum, htid, and probability
trimming_info = pd.DataFrame({
    'pagenum': pages['pagenum'],
    'wordcount': pages['nwords'],
    'htid': htids,
    'probabilities': probabilities[:, 1]  # Assuming the second column contains the probabilities for 'text'
})

# Now we define a simple function that uses predicted probabilities
# for pages to infer a good start page, where paratext gives way to text,
# and a good end page, where text gives way to paratext.

# In reality, text and paratext can be interleaved throughout a volume,
# but the odds of making an error, and destroying continuity, are fairly
# high in the middle of a volume. By contrast, the mere fact of position
# at the beginning or end of a volume is a strong enough signal to support
# a reliable, conservative "trimming" approach.

# Our goal is to stop trimming when we hit a
# sequence of three pages that are all text. For this reason, we combine
# the probability of the current page with the average probability of the
# next two pages, and use that as a guide. The default parameters
# set below have been determined by trial and error.

def simple_probabilistic_cut(vol_df, threshold = 0.5, longlookweight = 0.1):
    ''' This function takes a dataframe representing a volume, and
    parameters that define a probability threshold, along with the weight
    to be assigned to a "look forward" at the next two pages. It returns
    
    '''

    
    inferred_labels = []
    text_probabilities = vol_df['probabilities'].tolist()
    text_lengths = vol_df['wordcount'].tolist()
    text_lengths = [1 for x in text_lengths] # we're treating all pages equally;
                                                #weighting them did not help
    firstcut = len(text_probabilities)
    for i in range(len(text_probabilities)):
        lookforward = i + 3
        if lookforward >= len(text_probabilities):
            lookforward = len(text_probabilities)
        # longlookforward = i + 5
        # if longlookforward >= len(text_probabilities):
        #     longlookforward = len(text_probabilities)
        # longavg = sum(text_probabilities[i:lookforward]) / len(text_probabilities[i:lookforward])
        longavg = sum([text_probabilities[j] * text_lengths[j] for j in range(i, lookforward)]) / sum(text_lengths[i:lookforward])
        shortavg = text_probabilities[i]
        textavg = (shortavg * (1 - longlookweight)) + (longavg * longlookweight)
        if textavg >= threshold:
            firstcut = i
            break

    lastcut = 0
    for i in range(len(text_probabilities), 0, -1):
        lookback = i - 3
        if lookback < 0:
            lookback = 0
        # longlookback = i - 5
        # if longlookback < 0:
        #     longlookback = 0
        longavg = sum([text_probabilities[j] * text_lengths[j] for j in range(lookback, i)]) / sum(text_lengths[lookback:i])
        shortavg = text_probabilities[i - 1]
        textavg = (shortavg * (1 - longlookweight)) + (longavg * longlookweight)
        if textavg >= threshold:
            lastcut = i
            break
    
    if firstcut < lastcut:
        inferred_labels = ['para'] * firstcut + ['text'] * (lastcut - firstcut) + ['para'] * (len(text_probabilities) - lastcut)
        assert len(inferred_labels) == len(text_probabilities)
    else:
        inferred_labels = ['para'] * len(text_probabilities)
    
    return inferred_labels

unique_htids = trimming_info['htid'].unique()

page_labels = dict()

for htid in unique_htids:
    vol_df = trimming_info[trimming_info['htid'] == htid]
    vol_df = vol_df.sort_values(by='pagenum')
    vol_df = vol_df.reset_index(drop=True)
    vol_df['inferred_labels'] = simple_probabilistic_cut(vol_df, threshold = 0.5, longlookweight = 0.1)
    page_labels[htid] = vol_df

del trimming_info

# Now we can actually trim the files
trimming_metadata = dict()
ctr = 0

for htid in unique_htids:
    vol_df = page_labels[htid]
    input_path = os.path.join(input_dir, htid + '.norm.txt')
    output_path = os.path.join(output_dir, htid + '.trim.txt')
    pages = []
    thispage = []
    with open(input_path, encoding = 'utf-8') as file:
        lines = file.readlines()
    for l in lines:
        if l.startswith('<pb>'):
            pages.append(thispage)
            thispage = []
        else:
            thispage.append(l)
    pages.append(thispage)

    trimmed_pages = []
    cut_wordcount = 0
    for i, row in vol_df.iterrows():
        if row['inferred_labels'] == 'text':
            trimmed_pages.append(pages[i])
        else:
            cut_wordcount += row['wordcount']
    
    total_words = sum(vol_df['wordcount'])
    if total_words > 0:
        pct_trim = cut_wordcount / total_words
    else:
        pct_trim = 0

    trimmed_pages, removed = header.remove_headers(trimmed_pages, romannumerals)

    header_words = sum([len(x.split()) for x in removed])
    if total_words > 0:
        pct_header = header_words / total_words
    else:
        pct_header = 0

    with open(output_path, mode = 'w', encoding = 'utf-8') as file:
        for p in trimmed_pages:
            for l in p:
                file.write(l)
            file.write('<pb>\n')
    
    trimming_metadata[htid] = {'total_words': total_words, 'pct_trim': pct_trim, 'pct_header': pct_header,
                                'remaining_words': total_words - (cut_wordcount + header_words)}
    
    ctr += 1
    if ctr % 100 == 0:
        print(ctr)


# Write metadata to a file
metadata_path = os.path.join(output_dir, 'trimming_metadata.tsv')
with open(metadata_path, mode = 'w', encoding = 'utf-8') as file:
    file.write('htid\ttotal_words\tpct_trim\tpct_header\tremaining_words\n')
    for htid, metadata in trimming_metadata.items():
        file.write(htid + '\t' + str(metadata['total_words']) + '\t' + str(metadata['pct_trim']) + '\t' + str(metadata['pct_header']) + '\t' + str(metadata['remaining_words']) + '\n') 

print('Done!')