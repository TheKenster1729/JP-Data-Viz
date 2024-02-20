import pandas as pd
from plotly.colors import n_colors, hex_to_rgb, convert_dict_colors_to_same_type
from PIL import ImageColor

class Color:
    def __init__(self):
        self.palette = {
            "Red 100": "2d0709",
            "Red 90": "520408",
            "Red 80": "750e13",
            "Red 70": "a2191f",
            "Red 60": "da1e28",
            "Red 50": "fa4d56",
            "Red 40": "ff8389",
            "Red 30": "ffb3b8",
            "Red 20": "ffd7d9",
            "Red 10": "fff1f1",
            "Magenta 100": "2a0a18",
            "Magenta 90": "510224",
            "Magenta 80": "740937",
            "Magenta 70": "9f1853",
            "Magenta 60": "d02670",
            "Magenta 50": "ee5396",
            "Magenta 40": "ff7eb6",
            "Magenta 30": "ffafd2",
            "Magenta 20": "ffdee8",
            "Magenta 10": "fff0f7",
            "Purple 100": "1c0f30",
            "Purple 90": "31135e",
            "Purple 80": "491d8b",
            "Purple 70": "6929c4",
            "Purple 60": "8a3ffc",
            "Purple 50": "a56eff",
            "Purple 40": "be95ff",
            "Purple 30": "d4bbff",
            "Purple 20": "e8daff",
            "Purple 10": "f6f2ff",
            "Blue 100": "001141",
            "Blue 90": "001d6c",
            "Blue 80": "002d9c",
            "Blue 70": "0043ce",
            "Blue 60": "0f62fe",
            "Blue 50": "4589ff",
            "Blue 40": "78a9ff",
            "Blue 30": "a6c8ff",
            "Blue 20": "d0e2ff",
            "Blue 10": "edf5ff",
            "Cyan 100": "061727",
            "Cyan 90": "012749",
            "Cyan 80": "003a6d",
            "Cyan 70": "00539a",
            "Cyan 60": "0072c3",
            "Cyan 50": "1192e8",
            "Cyan 40": "33b1ff",
            "Cyan 30": "82cfff",
            "Cyan 20": "bae6ff",
            "Cyan 10": "e5f6ff",
            "Teal 100": "081a1c",
            "Teal 90": "022b30",
            "Teal 80": "004144",
            "Teal 70": "005d5d",
            "Teal 60": "007d79",
            "Teal 50": "009d9a",
            "Teal 40": "08bdda",
            "Teal 30": "3ddbd9",
            "Teal 20": "9ef0f0",
            "Teal 10": "d9f9fb",
            "Green 100": "071908",
            "Green 90": "022d0d",
            "Green 80": "044317",
            "Green 70": "0e6027",
            "Green 60": "198038",
            "Green 50": "24a148",
            "Green 40": "42be65",
            "Green 30": "6fdc8c",
            "Green 20": "a7f0ba",
            "Green 10": "defbec",
            "Cool Gray 100": "121619",
            "Cool Gray 90": "21272a",
            "Cool Gray 80": "343a3f",
            "Cool Gray 70": "4d5358",
            "Cool Gray 60": "697077",
            "Cool Gray 50": "878d96",
            "Cool Gray 40": "a2a9b0",
            "Cool Gray 30": "c1c7cd",
            "Cool Gray 20": "dde1e6",
            "Cool Gray 10": "f2f4f8",
            "Gray 100": "161616",
            "Gray 90": "262626",
            "Gray 80": "393939",
            "Gray 70": "525252",
            "Gray 60": "6f6f6f",
            "Gray 50": "8d8d8d",
            "Gray 40": "a8a8a8",
            "Gray 30": "c6c6c6",
            "Gray 20": "e0e0e0",
            "Gray 10": "f4f4f4",
            "Warm Gray 100": "171414",
            "Warm Gray 90": "272525",
            "Warm Gray 80": "3c3838",
            "Warm Gray 70": "565151",
            "Warm Gray 60": "726e6e",
            "Warm Gray 50": "8f8b8b",
            "Warm Gray 40": "ada8a8",
            "Warm Gray 30": "cac5c4",
            "Warm Gray 20": "e5e0df",
            "Warm Gray 10": "f7f3f2"
            }
        self.hex_palette = {key: '#' + value for key, value in self.palette.items()}
        # self.region_colors = {"GLB": "rgb(127, 127, 127)", "USA": "rgb(0, 83, 154)", "CAN": "rgb(162, 25, 31)", "MEX": "rgb(4, 67, 23)", "JPN": "rgb(250, 77, 86)",
        #                       "ANZ": "rgb(0, 67, 206)", "EUR": "rgb(255, 221, 0)", "ROE": "rgb(61, 219, 217)", "RUS": "rgb(1, 39, 73)", "ASI": "rgb(73, 29, 139)",
        #                       "CHN": "rgb(82, 4, 8)", "IND": "rgb(245, 222, 179)", "BRA": "rgb(111, 220, 140)", "AFR": "rgb(166, 200, 255)", "MES": "rgb(212, 187, 255)", 
        #                       "LAM": "rgb(0, 65, 68)", "REA": "rgb(130, 207, 255)", "KOR": "rgb(0, 93, 93)", "IDZ": "rgb(114, 110, 110)"}
        self.region_colors = {"GLB": "#7F7F7F", "USA": "#5492C5", "CAN": "#1D4971", "MEX": "#80CDDF", "JPN": "#6E37A3",
                              "ANZ": "#1B344A", "EUR": "#679C82", "ROE": "#91C96E", "RUS": "#2B4739", "ASI": "#493B82",
                              "CHN": "#725D7A", "IND": "#979576", "BRA": "#16824D", "AFR": "#1A5A2D", "MES": "#D6D092", 
                              "LAM": "#38A8A3", "REA": "#CCBE2C", "KOR": "#52CE02", "IDZ": "#B03AC2"}
        self.scenario_colors = {"15C_med": "#750e13", "15C_opt": "#ffb3b8", "About15C_pes": "#740937", "About15C_med": "#ffafd2",
                                "About15C_opt": "#491d8b", "2C_pes": "#d4bbff", "2C_med": "#002d9c", "2C_opt": "#a6c8ff",
                                "Above2C_pes": "#003a6d", "Above2C_med": "#82cfff", "Above2C_opt": "#004144", "Ref": "#3ddbd9"}
        self.scenario_markers = {"Ref": "solid", "Above2C_med": "dot", "2C_med": "dash"}
        self.histogram_patterns = {"Ref": "", "2C": "/"}
        self.parallel_coords_colors = ["#785EF0", "#FFB000"]

    def generate_palette(self, n):
        rgb_colors = convert_dict_colors_to_same_type(self.hex_palette)
        return n_colors(rgb_colors["Red 80"], rgb_colors["Cool Gray 50"], n, colortype = "rgb")
    
    def convert_to_fill(self, color, alpha = 0.3):
        # add alpha channel
        rgb = ImageColor.getcolor(color, "RGB")
        rgba = tuple(list(rgb) + [alpha])
        return "rgba" + str(rgba)
    
    def lighten_hex(self, hex_color, brightness_offset = 1):
        if len(hex_color) != 7:
            raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
        rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
        new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
        new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
        # hex() produces "0x88", we want just "88"
        return "#" + "".join([hex(i)[2:] for i in new_rgb_int])

class Readability:
    def __init__(self):
        self.naming_df = pd.read_csv(r"display_names.csv")
        self.naming_dict_long_names_first = {i["Full Output Name"]:i["Display Name"] for i in self.naming_df.to_dict("records")}
        self.naming_dict_display_names_first = {v:k for k, v in self.naming_dict_long_names_first.items()}

class Options:
    def __init__(self):
        self.region_names = ["GLB", "USA", "CAN", "MEX", "JPN", "ANZ", "EUR", "ROE", "RUS", "ASI", "CHN", "IND", 
                                "BRA", "AFR", "MES", "LAM", "REA", "KOR", "IDZ"]
        self.scenarios = ['15C_med', '15C_opt', 'About15C_pes', 'About15C_med', 'About15C_opt','2C_pes', '2C_med', '2C_opt', 'Above2C_pes', 'Above2C_med', 'Above2C_opt', 'Ref']
        self.scenario_display_names = {"15C_med": "1.5C Med", "15C_opt": "1.5C Opt", "2C_med": "2C Med", "2C_opt": "2C Opt", "2C_pes": "2C Pes", "About15C_opt": "About 1.5C Opt",
                                       "About15C_pes": "About 1.5C Pes", "Above2C_med": "Above 2C Med", "Above2C_opt": "Above 2C Opt", "Above2C_pes": "Above 2C Pes", "Ref": "Ref"}
        self.scenario_display_names_rev = {v:k for k, v in self.scenario_display_names.items()}
        self.outputs = Readability().naming_dict_long_names_first.keys()
        self.years = [i for i in range(2020, 2101, 5)]
        self.markers = {"Ref": "circle", "2C": "triangle-up-open-dot"}
        self.input_names = [
                    "E-LK",
                    "E-NE(final)",
                    "E-NOE",
                    "ESUBE(EL/EI)",
                    "ESUBE(oth)",
                    "LK AGRI",
                    "LK ENOE",
                    "LK ELEC",
                    "LK EINT",
                    "LK SERV",
                    "LK OTHR",
                    "LK TRANS",
                    "LK DWE",
                    "LK FOOD",
                    "FF-COAL",
                    "FF-OIL",
                    "FF-GAS",
                    "VINT",
                    "AEEI USA",
                    "AEEI CAN",
                    "AEEI MEX",
                    "AEEI JPN",
                    "AEEI ANZ",
                    "AEEI EUR",
                    "AEEI ROE",
                    "AEEI RUS",
                    "AEEI ASI",
                    "AEEI KOR",
                    "AEEI IDZ",
                    "AEEI CHN",
                    "AEEI IND",
                    "AEEI BRA",
                    "AEEI AFR",
                    "AEEI MES",
                    "AEEI LAM",
                    "AEEI REA",
                    "oil",
                    "gas",
                    "coal",
                    "NGCC",
                    "PC",
                    "Nuclear",
                    "PV",
                    "wind",
                    "Bio",
                    "NGCAP",
                    "PCCAP",
                    "BioCCS",
                    "WindGas",
                    "WindBio",
                    "Biol-Oil",
                    "GDP",
                    "Pop",
                    "USA GDP",
                    "Non-USA GDP",
                    "USA Pop",
                    "Non-USA Pop",
                    "CHN GDP",
                    "Non-CHN GDP",
                    "CHN Pop",
                    "Non-CHN Pop",
                    "EUR GDP",
                    "Non-EUR GDP",
                    "EUR Pop",
                    "Non-EUR Pop",
                    "CAN GDP",
                    "Non-CAN GDP",
                    "CAN Pop",
                    "Non-CAN Pop",
                    "MEX GDP",
                    "Non-MEX GDP",
                    "MEX Pop",
                    "Non-MEX Pop",
                    "JPN GDP",
                    "Non-JPN GDP",
                    "JPN Pop",
                    "Non-JPN Pop",
                    "ANZ GDP",
                    "Non-ANZ GDP",
                    "ANZ Pop",
                    "Non-ANZ Pop",
                    "ROE GDP",
                    "Non-ROE GDP",
                    "ROE Pop",
                    "Non-ROE Pop",
                    "RUS GDP",
                    "Non-RUS GDP",
                    "RUS Pop",
                    "Non-RUS Pop",
                    "ASI GDP",
                    "Non-ASI GDP",
                    "ASI Pop",
                    "Non-ASI Pop",
                    "IND GDP",
                    "Non-IND GDP",
                    "IND Pop",
                    "Non-IND Pop",
                    "BRA GDP",
                    "Non-BRA GDP",
                    "BRA Pop",
                    "Non-BRA Pop",
                    "AFR GDP",
                    "Non-AFR GDP",
                    "AFR Pop",
                    "Non-AFR Pop",
                    "MES GDP",
                    "Non-MES GDP",
                    "MES Pop",
                    "Non-MES Pop",
                    "LAM GDP",
                    "Non-LAM GDP",
                    "LAM Pop",
                    "Non-LAM Pop",
                    "REA GDP",
                    "Non-REA GDP",
                    "REA Pop",
                    "Non-REA Pop",
                    "KOR GDP",
                    "Non-KOR GDP",
                    "KOR Pop",
                    "Non-KOR Pop",
                    "IDZ GDP",
                    "Non-IDZ GDP",
                    "IDZ Pop",
                    "Non-IDZ Pop"
                ]
        # in case
        # name = str(bound) + {1: 'st', 2: 'nd', 3: 'rd'}.get(4 if 10 <= bound % 100 < 20 else bound % 10, "th")
        
if __name__ == "__main__":
    region_colors = {"GLB": "#7F7F7F", "USA": "#5492C5", "CAN": "#1D4971", "MEX": "#80CDDF", "JPN": "#6E37A3",
                            "ANZ": "#1B344A", "EUR": "#679C82", "ROE": "#91C96E", "RUS": "#2B4739", "ASI": "#493B82",
                            "CHN": "#725D7A", "IND": "#979576", "BRA": "#16824D", "AFR": "#1A5A2D", "MES": "#D6D092", 
                            "LAM": "#38A8A3", "REA": "#CCBE2C", "KOR": "#52CE02", "IDZ": "#B03AC2"}
    print(list(region_colors.values()))