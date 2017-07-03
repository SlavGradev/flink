#!/usr/bin/env python

"""
====================
Horizontal bar chart
====================

This example showcases a simple horizontal bar chart.
"""
import matplotlib.pyplot as plt
from operator import itemgetter
plt.rcdefaults()
import numpy as np
import sys
from matplotlib.backends.backend_pdf import PdfPages


nanosecond_in_seconds = 10**-9

f = open(sys.argv[1], 'r')

plt.rcdefaults()
fig, ax = plt.subplots()

start_times = {}
end_times = {}
t_0 = 0

for line in f:
	tokens = line.split('|')
	task_name = tokens[0]
	task_time = tokens[1].split(" : ")[1]
	if "starts" in tokens[1]:
		start_times[task_name] = task_time
		if "DataSource" in task_name:
			t_0 = float(task_time)
	else:
		end_times[task_name] = task_time 

events = []


for task, start_time in start_times.iteritems():
	end_time = end_times[task]
	events.append((task, (float(start_time) - t_0) * nanosecond_in_seconds, (float(end_time) - t_0) * nanosecond_in_seconds ))

events.sort(key=itemgetter(1))

for e in events:
	print e
	
# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(events) - 2)

ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

pp = PdfPages('_graph.pdf')

pp.savefig()

pp.close()

