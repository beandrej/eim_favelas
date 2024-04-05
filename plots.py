import matplotlib.pylab as plt

energy_sources = ['Solar', 'HP']
production_values = [985, 15]
explode = (0, 0)
colors = ['#1f77b4', '#ff7f0e']

# Creating the pie plot
plt.figure(figsize=(10, 10))
plt.pie(production_values, autopct='%1.1f%%', explode=explode, startangle=40, pctdistance=0.9, colors=colors)
plt.title('Energy production')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(loc='upper left', labels=energy_sources)
plt.savefig('plots/min_em_prod',dpi=300)
plt.show()