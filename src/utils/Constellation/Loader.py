import cudf


def LoadDataset():
    # for faster loading, use GPU DF
    plot_df_3d = cudf.read_parquet("./datasets/umapped3D.parquet", engine='pyarrow')
    plot_df_2d = cudf.read_parquet("./datasets/umapped2D.parquet", engine='pyarrow')

    # then move to CPU
    return plot_df_3d.to_pandas(), plot_df_2d.to_pandas()