import os
import pandas as pd
import numpy as np

from config import DatasetCreatorSettings
from dataset_preprocessor import DatasetPreprocessor

settings = DatasetCreatorSettings()

NAN_THRESHOLD = 5

if __name__ == "__main__":

    preprcessor = DatasetPreprocessor(
        source_data_dir=settings.source_data_dir,
        to_save_data_dir=settings.save_dataset_path,
    )

    # load from source
    csv_paths = preprcessor.get_csv_paths()

    attribute_dfs = []
    for path in csv_paths:
        data = pd.read_csv(path, skiprows=4)

        unnamed_columns = preprcessor.find_non_unnamend_cols(data)

        if unnamed_columns:
            data.drop(columns=unnamed_columns, inplace=True)

        # TODO: nastav treshold tak, aby ukazoval počet nezaznamenaných hodnôt v riadku daného indikátoru -> ešte pred melt

        melted_df = preprcessor.melt_single_df(
            df=data,
            columns_to_keep=[
                "Country Name",
                "Country Code",
                "Indicator Name",
                "Indicator Code",
            ],
        )
        melted_df.drop(columns=["Indicator Name", "Indicator Code"], inplace=True)

        attribute_dfs.append(melted_df)

    # merged_df = pd.concat(attribute_dfs, axis=0)
    merged_df = attribute_dfs[0]

    for df in attribute_dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on=["Country Name", "Country Code", "Year"],
            how="inner",
        )

    merged_df["NaN_count"] = merged_df.isna().sum(axis=1)
    sorted_merged_df = merged_df.sort_values(by="NaN_count", ascending=True)

    # print(sorted_merged_df.head(50))

    # print(sorted_merged_df.tail(50))

    # play with the number of occurences
    threshold_nan_df = sorted_merged_df[sorted_merged_df["NaN_count"] <= NAN_THRESHOLD]

    print(threshold_nan_df.shape)

    # TODO: zisti, ktoré stĺpce chýbajú

    threshold_nan_df.to_csv("test.csv", index=False)
