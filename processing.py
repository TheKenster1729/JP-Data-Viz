# goal of this module is to take in data from every source and return data that can be used in the analysis and figure modules
# modularity is key - if data sources change, only this module needs to be updated, not any others
import os
import pandas as pd
import openpyxl as op
import numpy as np

class LoadData:
    def __init__(self, **filters):
        self.directory = r".\Cleaned Data"
        self.filters = filters
        self.regions = self.filters["region"]
        self.output_file = self.filters["output_name"] + ".csv"
        self.scenario = self.filters["scenario"]
        self.years = self.filters["year"]
        self.path_to_file = os.path.join(self.directory, self.scenario, self.output_file)
        
    def csv_to_dataframe(self):
        self.dataframe = pd.read_csv(self.path_to_file)
        # first by region
        self.filtered_df = self.dataframe.query("Region == @self.regions")
        # then by year
        self.filtered_df = self.filtered_df.query("Year == @self.years")
        
        return self.filtered_df

    def export_dataframe(self):
        pass

class Scripts:
    """
    This class contains functions that need only be run once. By running this file itself, with complete datasets in
    the current working directory, these scripts will run and create/populate a "Cleaned Data" folder.

    These scripts convert the many different Excel spreadsheets for each output distribution under Reference and Policy cases
    to one .csv file per case-output pair in "long-form". The advantages of this approach are 1) more clarity and less complexity in the number and 
    names of files, and 2) it is faster for Pandas to load .csv files, even large ones, than Excel files.
    """
    def __init__(self, initial_directory = r".\Raw Data", output_directory = r".\Cleaned Data"):
        self.initial_directory = initial_directory
        self.output_directory = output_directory
        self.aggregate_df = None
        self.cases = ["Ref"]
        # self.list_filenames_by_case_dict = self.list_filenames_by_case()
        self.main()

    def pre_process_spreadsheet(self, workbook, worksheet, delete_start_column, delete_number, file_path):
        workbook[worksheet].delete_cols(int(delete_start_column), int(delete_number))
        workbook.save(file_path)

    # returns a dict of {case: [filenames]}
    def list_filenames_by_case(self):
        # reference first
        ref_files_path = os.path.join(self.initial_directory, "Ref")
        ref_files = os.listdir(ref_files_path)

        # policy next
        policy_files_path = os.path.join(self.initial_directory, "2C")
        pol_files = os.listdir(policy_files_path)

        filenames_by_case_dict = {"Ref": ref_files, "2C": pol_files}

        return filenames_by_case_dict

    def collect_data_from_one_file(self, filename, path_to_file, sheet_name):
        location = os.path.join(path_to_file, filename)
        region = sheet_name.split('_')[-1]
        value_name = '_'.join(filename.split('_')[1:-1])

        # construct wide-form df
        df = pd.read_excel(location, sheet_name = sheet_name).dropna(how = "all", axis = 1)
        df = df.set_index(df.columns[0])
        melted_df = df.melt(var_name = "Year", value_name = value_name)

        # somewhat verbose way of constructing the longform df
        longform_df = pd.DataFrame()
        longform_df["Run #"] = np.tile(df.index, len(df.columns))
        longform_df["Region"] = [region]*len(melted_df)
        longform_df["Year"] = melted_df["Year"]
        longform_df["Output Name"] = [value_name]*len(melted_df)
        longform_df["Value"] = melted_df[value_name]

        return longform_df

    def aggregate_case_data(self, df_to_add, aggregate_df):
        return pd.concat([aggregate_df, df_to_add])

    def export_to_long_form_csv(self):
        pass

    def main(self):
        for case in self.cases:
            case_folder_path = os.path.join(self.initial_directory, case)
            files = os.listdir(case_folder_path)
            for file in files[21:]:
                self.aggregate_df = pd.DataFrame(columns = ["Run #", "Region", "Year", "Output Name", "Value"])
                file_path = os.path.join(case_folder_path, file)
                workbook = op.load_workbook(file_path)
                delete_start_column = input("Enter starting column to delete for {}:".format(file))
                delete_number = input("Enter the number of columns to delete for {}:".format(file))
                for worksheet in workbook.sheetnames[1:]:
                    self.pre_process_spreadsheet(workbook, worksheet, delete_start_column, delete_number, file_path)
                    df_to_add = self.collect_data_from_one_file(file, case_folder_path, worksheet)
                    self.aggregate_df = self.aggregate_case_data(df_to_add, self.aggregate_df)
                save_path = os.path.join(self.output_directory, case, "{}.csv".format('_'.join(file.split('_')[1:-1])))
                self.aggregate_df.to_csv(save_path)

if __name__ == "__main__":
    Scripts()