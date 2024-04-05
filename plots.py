import matplotlib.pylab as plt

energy_sources = ['Solar', 'HP', 'Battery']
production_values = [360000, 804, 70000]


# Creating the pie plot
plt.figure(figsize=(10, 10))
plt.pie(production_values, autopct='%1.1f%%', startangle=40, pctdistance=0.9, labels=['360000 kW', '804 kW', '70000 kW'])
plt.title('Capacity installed')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(loc='upper left', labels=energy_sources)
plt.savefig('plots/min_em_cap2',dpi=300)
plt.show()