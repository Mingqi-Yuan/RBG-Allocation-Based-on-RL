import numpy as np
import matplotlib.pyplot as plt 

better_avg_percent = np.load('better_avg_percent.npy')
better_worst_percent = np.load('better_worst_percent.npy')

sorted_avg = sorted([x+1 for x in better_avg_percent])
sorted_worst = sorted([x+1 for x in better_worst_percent])

prop = np.arange(1, len(sorted_avg)+1) / len(sorted_avg)

plt.title('CDF Analysis')
plt.plot(sorted_avg, prop, color = 'red', linewidth = 1, alpha = 1, label = 'Average User Rate Ratio')
plt.plot(sorted_worst, prop, color = 'blue', linewidth = 1, alpha = 1, label = 'Worst User Rate Ratio')
plt.legend()

plt.xlabel('Data Rate Improvement Ratio')
plt.grid()
plt.show()