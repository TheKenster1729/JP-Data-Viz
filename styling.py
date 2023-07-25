class Color:
    def __init__(self):
        self.timeseries_colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

class Readability:
    def __init__(self):
        pass

class Options:
    def __init__(self):
        self.region_names = ["GLB", "USA", "CAN", "MEX", "JPN", "ANZ", "EUR", "ROE", "RUS", "ASI", "CHN", "IND", 
                                "BRA", "AFR", "MES", "LAM", "REA", "KOR", "IDZ"]
        self.scenarios = ["Ref", "2C"]
        self.outputs = ["GDP_billion_USD2007", "population_million_people", "primary_energy_use_Biomass_EJ"]
        self.years = [i for i in range(2020, 2101, 5)]