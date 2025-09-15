import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import hdbscan
import umap.umap_ as umap

df = pd.read_parquet('nutritional_complete_imputed.parquet', engine='pyarrow')
print(df.head(10))

names = df['food_name'].fillna('')
names_1 = df['food_type_1'].fillna('')
names_2 = df['food_type_2'].fillna('')

 
names = names.loc[df.index]
names_1 = names_1.loc[df.index]
names_2 = names_2.loc[df.index]

 
numeric_data = df.select_dtypes(include=['float64', 'int64']).fillna(0)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(numeric_data)
 
combined_text = names + ' ' + names_1 + ' ' + names_2
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(combined_text)
 
combined_features = hstack([csr_matrix(df_scaled), tfidf_matrix])
 
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=50,
    min_dist=0.4,
    n_jobs=8,
    n_epochs=500,
    learning_rate=1.0,
    random_state=42,
    spread=3.0
)
umap_result = umap_model.fit_transform(combined_features)
 
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
labels = clusterer.fit_predict(umap_result)
 
plot_df = pd.DataFrame({
    'UMAP1': umap_result[:, 0],
    'UMAP2': umap_result[:, 1],
    'name': names.values,
    'food_type_1': names_1.values,
    'food_type_2': names_2.values,
    'cluster': labels
})
 
for col in numeric_data.columns:
    plot_df[col] = df[col].values

print(plot_df.head(10))

plot_df.to_parquet("umapped2D.parquet", engine='pyarrow')

print(len(df))
