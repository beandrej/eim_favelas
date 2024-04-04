import os
import pandas as pd

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

    elec = year["Demand (MWh)"] * 0.033478 # --- Scaling factor

    heat = pd.read_csv('brazil_heat.csv', delimiter=',', comment='#')['total_demand']

    assert len(elec) == 8760 and len(heat) == 8760
    return elec, heat
