import os
import pandas as pd
import numpy as np

def get_data():
    dir = 'elec_demand'
    df_months = []
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            path = os.path.join(dir, filename)
            df = pd.read_csv(path)
            df.drop(df.index[-1], inplace=True)
            df_months.append(df)

    year = pd.concat(df_months, ignore_index=True)
    year.dropna(inplace=True)

    scaling_factor = (0.1741*55361*0.25*12)/year["Demand (MWh)"].sum() # monthly conusmption per household * number of households * 0.25 * 12 months / total consumption

    elec = year["Demand (MWh)"] * scaling_factor *1000 # in kWh

    heat = pd.read_csv('brazil_heat.csv', delimiter=',', comment='#')['total_demand']
    heat = heat * 1000 # in kWh

    assert len(elec) == 8760 and len(heat) == 8760
    return np.array(elec), np.array(heat)
