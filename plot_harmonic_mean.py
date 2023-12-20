#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def harmonic_mean(a, b):
	if a == 0 or b == 0:
		return float('NaN')
	return 2/(1/a + 1/b)

if __name__ == "__main__":
	value_A = [0.2, 0.5, 1.0, 2.5, 5.0, 10.0]
	value_B = np.linspace(0.1, 20, num=400)

	for a in value_A:
		avg = []
		for b in value_B:
			avg.append(harmonic_mean(a, b))
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(value_B, avg)
	plt.show()
