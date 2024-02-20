import geopandas
import matplotlib.pyplot as plt

df = geopandas.read_file("eppa6_regions.shp")
print(df.head())