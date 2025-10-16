import cudf


def LoadDataset():
    plot_df_3d = cudf.read_parquet("./datasets/umapped3D.parquet", engine='pyarrow')
    plot_df_2d = cudf.read_parquet("./datasets/umapped2D.parquet", engine='pyarrow')

    return plot_df_3d, plot_df_2d