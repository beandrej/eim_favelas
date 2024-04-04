import pandas as pd

def get_data():
    jan = pd.read_csv('elec_demand/jan.csv')
    feb = pd.read_csv('elec_demand/feb.csv')
    mar = pd.read_csv('elec_demand/mar.csv')
    apr = pd.read_csv('elec_demand/apr.csv')
    mai = pd.read_csv('elec_demand/mai.csv')
    jun = pd.read_csv('elec_demand/jun.csv')
    jul = pd.read_csv('elec_demand/jul.csv')
    aug = pd.read_csv('elec_demand/aug.csv')
    sep = pd.read_csv('elec_demand/sep.csv')
    okt = pd.read_csv('elec_demand/okt.csv')
    nov = pd.read_csv('elec_demand/nov.csv')
    dez = pd.read_csv('elec_demand/dez.csv')
    df_months = [jan, feb, mar, apr, mai, jun, jul, aug, sep, okt, nov, dez]

    for i, df in enumerate(df_months):
        df_months[i] = df.drop(df.index[-1])
    year = pd.concat(df_months, ignore_index=True)
    year.dropna(inplace=True)

    elec = year["Demand (MWh)"] * 0.033478 # --- Scaling factor

    heat = pd.read_csv('brazil_heat.csv', delimiter=',', comment='#')['total_demand']

    assert len(elec) == 8760 and len(heat) == 8760
    return elec, heat
