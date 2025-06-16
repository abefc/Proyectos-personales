# -*- coding: utf-8 -*-
"""
Para este primer caso de estudio he utilizado el script desarrollado por Jorge Casillas
proporcionado como ejemplo en prado, realizando modificaciones para que se ajuste a los requerimientos
de la práctica.

Caso de estudio 1:
  Alojamientos del barrio "Ronda" con Rating > 0

Algoritmos de clustering aplicados:
  - K-Means
  - Jerárquico (AgglomerativeClustering)
  - DBSCAN
  - MeanShift
  - SpectralClustering

Se calculan métricas, se generan visualizaciones y se exploran parámetros.

"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn import metrics
from sklearn.impute import KNNImputer
from math import floor
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS

def norm_to_zero_one(df):
    return (df - df.min()) / (df.max() - df.min())

# Carga de datos
datos = pd.read_csv('alojamientos_booking_Granada_2024.csv', sep=';', encoding="iso-8859-1")
# Imputación de superficie
media_superficie = datos.loc[datos["Surface Area (m2)"] > 0, "Surface Area (m2)"].mean()
datos.loc[datos["Surface Area (m2)"] == 0, "Surface Area (m2)"] = media_superficie
# Caso de estudio: barrio Ronda, Rating > 0
subset = datos[datos.Location.str.contains('Ronda', case=False, na=False) & (datos.Rating > 0)]

# Renombrar columnas a nombres más sencillos
subset = subset.rename(columns={
    "Price avg": "precio", 
    "Distance": "distancia", 
    "Ranking position avg": "posicion", 
    "Guests": "huespedes",
    "Surface Area (m2)": "superficie"
})

usadas = ['precio', 'distancia', 'posicion', 'huespedes', 'superficie']
X = subset[usadas]

# Normalización
X_normal = X.apply(norm_to_zero_one)

###############################################################################
# Función auxiliar para calcular métricas, imprimir resultados y mostrar tamaños
###############################################################################
def evaluar_clustering(X_normal, labels, nombre_alg):
    metric_CH = metrics.calinski_harabasz_score(X_normal, labels)
    muestra_silhouette = 0.2 if (len(X_normal) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_normal, labels, metric='euclidean', 
                                         sample_size=floor(muestra_silhouette*len(X_normal)), random_state=123456)
    print("\n--- Resultados {} ---".format(nombre_alg))
    print("Calinski-Harabasz Index: {:.3f}".format(metric_CH))
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

    clusters_df = pd.DataFrame(labels, index=X_normal.index, columns=['cluster'])
    size = clusters_df['cluster'].value_counts().sort_index()
    print("Tamaño de cada cluster:")
    for i, c in size.items():
        print('%s: %5d (%5.2f%%)' % (i, c, 100*c/len(clusters_df)))
    return clusters_df

def scatter_matrix(X, labels, nombre, variables, filename):
    # Generar scatter matrix coloreado por cluster
    data_plot = pd.concat([X, pd.DataFrame(labels, index=X.index, columns=['cluster'])], axis=1)
    sns.set()
    k = len(np.unique(labels))
    colors = sns.color_palette("Paired", n_colors=k)
    # Reajuste de etiquetas de clusters negativos (para el DBSCAN)
    data_plot['cluster'] = data_plot['cluster'] + data_plot['cluster'].min().astype(int)*(-1) # Para DBSCAN, si hay -1, ajustamos
    plt.figure()
    pp = sns.pairplot(data_plot, vars=variables, hue="cluster", palette=colors, diag_kind="hist", plot_kws={"s": 25})
    pp.fig.subplots_adjust(wspace=.03, hspace=.03)
    pp.fig.set_size_inches(15, 15)
    pp.fig.suptitle(nombre, y=1.02)
    pp.savefig(filename)
    plt.close(pp.fig)
def histogram_cluster_distribution(data, labels, variables, output_file):
    data_plot = pd.concat([data, pd.DataFrame(labels, index=data.index, columns=['cluster'])], axis=1)

    num_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(num_clusters, len(variables), figsize=(15, num_clusters * 3), sharex='col', sharey='row')

    for cluster_id in range(num_clusters):
        cluster_data = data_plot[data_plot['cluster'] == cluster_id]
        for var_idx, variable in enumerate(variables):
            ax = axes[cluster_id, var_idx]
            sns.histplot(
                cluster_data[variable],
                kde=True,
                ax=ax,
                color=f"C{cluster_id}",
                bins=20
            )
            if cluster_id == 0:
                ax.set_title(variable, fontsize=10)
            if var_idx == 0:
                ax.set_ylabel(f"Cluster {cluster_id + 1}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)

    # Ajustar el layout
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
###############################################################################
# 1. K-means con k=4 (ya estaba hecho, pero reorganizo un poco)
###############################################################################
print("----- Ejecutando K-Means con k=4 -----")
t = time.time()
k_means = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=123456)
cluster_predict = k_means.fit_predict(X_normal)
tiempo = time.time() - t
print("Tiempo: {:.2f}s".format(tiempo))

clusters_kmeans = evaluar_clustering(X_normal, cluster_predict, "K-Means (k=4)")

# Heatmap de centroides K-Means
centers = pd.DataFrame(k_means.cluster_centers_, columns=list(X))

# Desnormalización de los valores de los centroides
centers_desnormal = centers.copy()
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

plt.figure()
# Los índices se mantienen como 0, 1, 2, 3
hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
hm.set_ylim(len(centers), 0)
hm.figure.set_size_inches(10, 10)
hm.figure.savefig("centroides_kmeans_nuevo.pdf")
plt.close()


# Scatter matrix para K-means
scatter_matrix(X, cluster_predict, "K-Means (k=4)", usadas, "scatter_kmeans.pdf")

# MDS para visualizar distancias entre centroides K-Means
dist_centers = metrics.pairwise_distances(k_means.cluster_centers_)
mds = MDS(n_components=2, random_state=123)
coords = mds.fit_transform(dist_centers)
plt.figure()
plt.scatter(coords[:,0], coords[:,1], s=300, c=range(len(coords)), cmap='Paired')
for i, (x_m, y_m) in enumerate(coords):
    plt.text(x_m, y_m, "C{}".format(i), fontsize=14, ha='center', va='center', color='black')
plt.title("MDS sobre centroides de K-Means")
plt.savefig("mds_centroides_kmeans.pdf")
plt.close()

# Cálculo de silhouette por muestra
sil_values = metrics.silhouette_samples(X_normal, cluster_predict)
df_sil = pd.DataFrame({'cluster': cluster_predict, 'silhouette': sil_values})

plt.figure()
sns.violinplot(x='cluster', y='silhouette', data=df_sil, inner='box', palette='Paired')
plt.title("Distribución del coeficiente de silhouette por cluster - K-Means (k=4)")
plt.savefig("violinplot_kmeans.pdf")
plt.close()

#pairplot
histogram_cluster_distribution(X, cluster_predict, usadas, "histogram_kmeans.png")

# Ejemplo de variación de parámetros en K-Means (elegir k óptimo)
print("\n--- Variando k en K-Means ---")
for k_test in [2,3,4,5,6]:
    km_test = KMeans(n_clusters=k_test, n_init=5, random_state=123456)
    pred_test = km_test.fit_predict(X_normal)
    ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
    sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
    print("k={}: CH={:.3f}, SC={:.5f}".format(k_test, ch_test, sc_test))


###############################################################################
# 2. Clustering Jerárquico (AgglomerativeClustering)
###############################################################################
print("\n----- Ejecutando Clustering Jerárquico (linkage='ward', n_clusters=4) -----")
hier = AgglomerativeClustering(n_clusters=4, linkage='ward')
t = time.time()
hier_predict = hier.fit_predict(X_normal)
tiempo = time.time() - t
print("Tiempo: {:.2f}s".format(tiempo))

clusters_hier = evaluar_clustering(X_normal, hier_predict, "Jerárquico (ward, n=4)")

# Dendrograma: para el dendrograma usamos scipy.linkage
Z = sch.linkage(X_normal, 'ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(Z, truncate_mode='level', p=6) 
plt.title("Dendrograma (Ward)")
plt.savefig("dendrograma_ward.pdf")
plt.close()

# Scatter matrix para Jerárquico
scatter_matrix(X, hier_predict, "Jerárquico (ward)", usadas, "scatter_hier_ward.pdf")

#histograma clusters
histogram_cluster_distribution(X, hier_predict, usadas, "histogram_jerarquico.png")


# Cálculo de silhouette por muestra
sil_values = metrics.silhouette_samples(X_normal, hier_predict)
df_sil = pd.DataFrame({'cluster': hier_predict, 'silhouette': sil_values})

plt.figure()
sns.violinplot(x='cluster', y='silhouette', data=df_sil, inner='box', palette='Paired')
plt.title("Distribución del coeficiente de silhouette por cluster - AgglomerativeClustering")
plt.savefig("violinplot_jerárquico.pdf")
plt.close()

# Ejemplo variando el método de enlace en jerárquico
print("\n--- Variando método de enlace en Jerárquico (n_clusters=4) ---")
for link_method in ['ward','complete','average']:
    hier_test = AgglomerativeClustering(n_clusters=4, linkage=link_method)
    pred_test = hier_test.fit_predict(X_normal)
    ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
    sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
    print("{}: CH={:.3f}, SC={:.5f}".format(link_method, ch_test, sc_test))

###############################################################################
# 3. DBSCAN
###############################################################################
print("\n----- Ejecutando DBSCAN (eps=0.25, min_samples=6) -----")
dbscan = DBSCAN(eps=0.25, min_samples=6)
t = time.time()
dbscan_predict = dbscan.fit_predict(X_normal)
tiempo = time.time() - t
print("Tiempo: {:.2f}s".format(tiempo))

# Nota: DBSCAN puede producir un cluster -1 para ruido. Comprobamos si hay clusters válidos.
if len(np.unique(dbscan_predict)) > 1:
    evaluar_clustering(X_normal, dbscan_predict, "DBSCAN (eps=0.25, min_samples=6)")
    scatter_matrix(X, dbscan_predict, "DBSCAN (eps=0.2,min_samples=10)", usadas, "scatter_dbscan.pdf")
else:
    print("DBSCAN no encontró clusters (todos son ruido). Ajustar parámetros.")

#histograma clusters
histogram_cluster_distribution(X, dbscan_predict, usadas, "histogram_dbscan.png")



# Ejemplo variando parámetros en DBSCAN
print("\n--- Variando eps en DBSCAN (min_samples=6) ---")
for eps_val in [0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.5]:
    db_test = DBSCAN(eps=eps_val, min_samples=6)
    pred_test = db_test.fit_predict(X_normal)
    if len(np.unique(pred_test)) > 1: # hay más de un cluster
        ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
        sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
        print("eps={}: Clusters={}, CH={:.3f}, SC={:.5f}".format(eps_val, len(np.unique(pred_test)), ch_test, sc_test))
    else:
        print("eps={}: todos ruido o un solo cluster".format(eps_val))


###############################################################################
# 4. MeanShift
###############################################################################
print("\n----- Ejecutando MeanShift (bandwidth=0.3) -----")
# Cuidado: MeanShift puede ser lento. Si el conjunto es grande, podrías reducirlo.
t = time.time()

# Inicialización de MeanShift con bandwidth=0.3
meanshift = MeanShift(bandwidth=0.3)
ms_predict = meanshift.fit_predict(X_normal)
tiempo = time.time() - t
print("Tiempo: {:.2f}s".format(tiempo))

# Evaluación del clustering
evaluar_clustering(X_normal, ms_predict, "MeanShift")

# Generación del Scatter Matrix y guardado en PDF
scatter_matrix(X, ms_predict, "MeanShift", usadas, "scatter_meanshift.pdf")

# Impresión de los centroides identificados por MeanShift
print("Centroides MeanShift (bandwidth=0.3):")
ms_centers = meanshift.cluster_centers_
print(ms_centers)

# Generación del histograma de distribución de clusters y guardado en PNG
histogram_cluster_distribution(X, ms_predict, usadas, "histogram_meanshift.png")


###############################################################################
# 5. Spectral Clustering
###############################################################################
print("\n----- Ejecutando SpectralClustering (n_clusters=4) -----")
t = time.time()
spectral = SpectralClustering(n_clusters=4, random_state=123)
spectral_predict = spectral.fit_predict(X_normal)
tiempo = time.time() - t
print("Tiempo: {:.2f}s".format(tiempo))

evaluar_clustering(X_normal, spectral_predict, "SpectralClustering (n=4)")
scatter_matrix(X, spectral_predict, "SpectralClustering (n=4)", usadas, "scatter_spectral.pdf")

#histograma clusters
histogram_cluster_distribution(X, spectral_predict, usadas, "histogram_spectral.png")



print("\nAnálisis completado para el caso de estudio del barrio de Ronda (Rating > 0).")
