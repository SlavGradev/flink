#!/usr/bin/env python

import sys
import os
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter

read_from = '/home/skg113/gpuflink/results'
write_to = '/home/skg113/gpuflink/graphs'

num_ratios = 12
num_repeated = 3

millisecond_in_nanoseconds = 10**6
nanosecond_in_milliseconds = 10**-6
nanosecond_in_seconds = 10**-9
millisecond_in_seconds = 10**-3

file_name = os.path.join(read_from, sys.argv[1])

f = open(file_name , 'r')

#Descriptions
f.readline()
f.readline()

raw_data = {'percentages': [],
        'whole_execution_time': []}

zipped = []

for i in xrange(num_ratios):
	percentage = 0
	whole_execution_times = []

	for j in xrange(num_repeated):
		tokens = f.readline().split(',')
		print tokens
		percentage = tokens[0]		
		whole_execution_times.append(float(tokens[2]) * millisecond_in_seconds)
	raw_data['percentages'].append(percentage)
	raw_data['whole_execution_time'].append(statistics.mean(whole_execution_times))
	
df = pd.DataFrame(raw_data, columns = ['percentages', 'whole_execution_time'])

df

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,5))

# Set the bar width
bar_width = 0.75

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['whole_execution_time']))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]


# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score data
        df['whole_execution_time'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='Whole Execution Time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='cyan')

# set the x ticks with names
plt.xticks(tick_pos, df['percentages'])

# Set the label and legends
ax1.set_ylabel("Time in Seconds", fontsize=10)
ax1.set_xlabel("Percentage of the job run on the gpu")

#ax1.set_yscale('log')

plt.legend(loc='upper left')
plt.title("Linear Regression 20 million points, 10 iterations, Average of 3 runs, 16 nodes each")

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

pp = PdfPages(os.path.join(write_to, sys.argv[1].split('.')[0] + '_detailed_graph.pdf'))

pp.savefig()

pp.close()

	
