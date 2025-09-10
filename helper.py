import pandas as pd
# import dask.dataframe as dd


# def scale_values(dask_dataframe: dd, column_name:str):
#     # Scaling
#     global_min = dask_dataframe[dask_dataframe.anomaly == 0][column_name].min().compute()
#     global_max = dask_dataframe[dask_dataframe.anomaly == 0][column_name].max().compute()

#     dask_dataframe[f"{column_name}_scaled"] = (dask_dataframe[column_name] - global_min) / (global_max - global_min)

#     return dask_dataframe


# def get_scaled_data_values(csv_path: str, column_name: str):
#     ddf = dd.read_csv(csv_path)

#     ddf = scale_values(ddf, column_name)
#     df = ddf.compute()

#     return df

        
def extract_signal_and_anomaly_array(df, column_name:str):
    data = df[f"{column_name}"].values
    labels = df["anomaly"].values
    return data, labels


def crop_datetime(df, start_datetime="", end_datetime="", print_data_info=False):

    if print_data_info:
        print(df.head())
        print(df.info())
        print(df.describe())

    start_dt = pd.to_datetime(start_datetime, utc=True)
    end_dt  = pd.to_datetime(end_datetime, utc=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        data_filtered = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        data_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    return data_filtered