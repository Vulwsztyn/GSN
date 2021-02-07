import matplotlib.pyplot as plt
import numpy as np
import pickle

rewards = pickle.load( open( "rewards.p", "rb" ) )

fig, ax = plt.subplots()
ax.plot(rewards, '-')
ax.set_xlabel('Liczba rozegranych gier')
ax.set_ylabel('Średnia otrzymanego rewardu')

plt.savefig('rewards.png')