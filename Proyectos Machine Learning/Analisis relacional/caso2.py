# -*- coding: utf-8 -*-
"""
Comparación dentro de un mismo caso de estudio entre dos subconjuntos:
 - Barrio Albaicín con Rating > 0
 - Barrio Centro con Rating > 0

La estructura es muy parecida a la del caso original con el barrio Ronda.
Se aplican los mismos algoritmos: K-Means, Jerárquico, DBSCAN, MeanShift, SpectralClustering
Se calculan métricas, se generan visualizaciones y se exploran parámetros.
Además, al final se hace una pequeña comparación visible entre los resultados de ambos barrios.
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from sklearn import metrics
from math import floor
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS

def norm_to_zero_one(df):
    return (df - df.min()) / (df.max() - df.min())

def evaluar_clustering(X_normal, labels, nombre_alg, barrio):
    if len(np.unique(labels)) < 2:
        # Si solo hay un cluster, no se puede calcular Silhouette
        print(f"\n--- Resultados {nombre_alg} - {barrio} ---")
        print("Solo se ha encontrado un cluster. No se puede calcular Silhouette ni CH.")
        clusters_df = pd.DataFrame(labels, index=X_normal.index, columns=['cluster'])
        size = clusters_df['cluster'].value_counts().sort_index()
        print("Tamaño de cada cluster:")
        for i, c in size.items():
            print('%s: %5d (%5.2f%%)' % (i, c, 100*c/len(clusters_df)))
        return clusters_df, None, None

    metric_CH = metrics.calinski_harabasz_score(X_normal, labels)
    muestra_silhouette = 0.2 if (len(X_normal) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_normal, labels, metric='euclidean',
                                         sample_size=floor(muestra_silhouette*len(X_normal)), random_state=123456)

    print(f"\n--- Resultados {nombre_alg} - {barrio} ---")
    print("Calinski-Harabasz Index: {:.3f}".format(metric_CH))
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

    clusters_df = pd.DataFrame(labels, index=X_normal.index, columns=['cluster'])
    size = clusters_df['cluster'].value_counts().sort_index()
    print("Tamaño de cada cluster:")
    for i, c in size.items():
        print('%s: %5d (%5.2f%%)' % (i, c, 100*c/len(clusters_df)))
    return clusters_df, metric_CH, metric_SC

def scatter_matrix(X, labels, nombre, variables, filename):
    data_plot = pd.concat([X, pd.DataFrame(labels, index=X.index, columns=['cluster'])], axis=1)
    sns.set()
    k = len(np.unique(labels))
    colors = sns.color_palette("Paired", n_colors=k)
    data_plot['cluster'] = data_plot['cluster'] + data_plot['cluster'].min().astype(int)*(-1)
    plt.figure()
    pp = sns.pairplot(data_plot, vars=variables, hue="cluster", palette=colors, diag_kind="hist", plot_kws={"s": 25})
    pp.fig.subplots_adjust(wspace=.03, hspace=.03)
    pp.fig.set_size_inches(15,15)
    pp.fig.suptitle(nombre, y=1.02)
    pp.savefig(filename)
    plt.close(pp.fig)

def violinplot_silhouette(X_normal, labels, algoritmo, barrio):
    # Solo calculamos si hay más de un cluster
    if len(np.unique(labels)) < 2:
        return
    sil_values = metrics.silhouette_samples(X_normal, labels)
    df_sil = pd.DataFrame({'cluster': labels, 'silhouette': sil_values})
    plt.figure()
    sns.violinplot(x='cluster', y='silhouette', data=df_sil, inner='box', palette='Paired')
    plt.title(f"Distribución del coeficiente de silhouette por cluster - {algoritmo} - {barrio}")
    plt.savefig(f"violinplot_{algoritmo.replace(' ', '_')}_{barrio}.pdf")
    plt.close()

###########################
# Carga de datos y preprocesamiento
###########################
datos = pd.read_csv('alojamientos_booking_Granada_2024.csv', sep=';', encoding="iso-8859-1")

# Imputación de superficie
media_superficie = datos.loc[datos["Surface Area (m2)"] > 0, "Surface Area (m2)"].mean()
datos.loc[datos["Surface Area (m2)"] == 0, "Surface Area (m2)"] = media_superficie

barrios = {
    "Albaicín": datos[(datos.Location.str.contains('Albaicín', case=False, na=False)) & (datos.Rating > 0)],
    "Centro": datos[(datos.Location.str.contains('Centro', case=False, na=False)) & (datos.Rating > 0)]
}

# Verificamos que las columnas existen en el dataset antes de continuar
columnas_originales = ["Price difference", "Distance", "Rating", "Ranking position avg", "Total Beds"]
usadas = ['precio','distancia','valoracion','posicion', 'dormitorios']

silhouette_kmeans = {}

for barrio, subset in barrios.items():
    print(f"\n\n********** Análisis del Barrio: {barrio} con Rating > 0 **********")

    # Comprobar si el subset está vacío
    if subset.empty:
        print(f"No hay datos para {barrio} con Rating > 0. Se omite este caso.")
        continue

    # Comprobar que existen las columnas originales antes de renombrar
    for col in columnas_originales:
        if col not in subset.columns:
            print(f"La columna {col} no existe en los datos para {barrio}. No se puede continuar.")
            continue

    subset = subset.rename(columns={
        "Price difference": "precio", 
        "Distance": "distancia", 
        "Rating": "valoracion", 
        "Ranking position avg": "posicion",
        "Total Beds": "dormitorios"
    })

    # Comprobar que el subset tiene todas las columnas renombradas necesarias
    for col in usadas:
        if col not in subset.columns:
            print(f"La columna {col} no existe en {barrio} después de renombrar. No se puede continuar.")
            continue

    X = subset[usadas]
    if X.empty:
        print(f"El subconjunto {barrio} no tiene datos tras la selección de variables. Se omite.")
        continue

    X_normal = X.apply(norm_to_zero_one)

    # K-Means con k=4
    print("----- Ejecutando K-Means con k=4 -----")
    t = time.time()
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=123456)
    cluster_predict = k_means.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    clusters_kmeans, ch_kmeans, sc_kmeans = evaluar_clustering(X_normal, cluster_predict, "K-Means (k=4)", barrio)

    if sc_kmeans is not None:
        silhouette_kmeans[barrio] = sc_kmeans

    centers = pd.DataFrame(k_means.cluster_centers_, columns=list(X))
    centers_desnormal = centers.copy()
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    plt.figure()
    centers.index += 1
    hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
    hm.set_ylim(len(centers), 0)
    hm.figure.set_size_inches(10,10)
    hm.figure.savefig(f"centroides_kmeans_{barrio}.pdf")
    centers.index -= 1
    plt.close()

    scatter_matrix(X, cluster_predict, f"K-Means (k=4) - {barrio}", usadas, f"scatter_kmeans_{barrio}.pdf")

    dist_centers = metrics.pairwise_distances(k_means.cluster_centers_)
    mds = MDS(n_components=2, random_state=123)
    coords = mds.fit_transform(dist_centers)
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], s=300, c=range(len(coords)), cmap='Paired')
    for i, (x_m, y_m) in enumerate(coords):
        plt.text(x_m, y_m, "C{}".format(i+1), fontsize=14, ha='center', va='center', color='black')
    plt.title(f"MDS sobre centroides de K-Means - {barrio}")
    plt.savefig(f"mds_centroides_kmeans_{barrio}.pdf")
    plt.close()

    violinplot_silhouette(X_normal, cluster_predict, "K-Means (k=4)", barrio)

    print("\n--- Variando k en K-Means ---")
    for k_test in [2,3,4,5,6]:
        km_test = KMeans(n_clusters=k_test, n_init=5, random_state=123456)
        pred_test = km_test.fit_predict(X_normal)
        if len(np.unique(pred_test)) > 1:
            ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
            sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
            print("k={}: CH={:.3f}, SC={:.5f}".format(k_test, ch_test, sc_test))
        else:
            print(f"k={k_test}: Solo un cluster. No se calcula SC ni CH.")

    # Jerárquico
    print(f"\n----- Ejecutando Clustering Jerárquico (linkage='ward', n_clusters=4) - {barrio} -----")
    hier = AgglomerativeClustering(n_clusters=4, linkage='ward')
    t = time.time()
    hier_predict = hier.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, hier_predict, "Jerárquico (ward, n=4)", barrio)

    Z = sch.linkage(X_normal, 'ward')
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z, truncate_mode='level', p=10)
    plt.title(f"Dendrograma (Ward) - {barrio}")
    plt.savefig(f"dendrograma_ward_{barrio}.pdf")
    plt.close()

    scatter_matrix(X, hier_predict, f"Jerárquico (ward) - {barrio}", usadas, f"scatter_hier_ward_{barrio}.pdf")

    violinplot_silhouette(X_normal, hier_predict, "Jerárquico (ward, n=4)", barrio)

    print("\n--- Variando método de enlace en Jerárquico (n_clusters=4) ---")
    for link_method in ['ward','complete','average']:
        hier_test = AgglomerativeClustering(n_clusters=4, linkage=link_method)
        pred_test = hier_test.fit_predict(X_normal)
        if len(np.unique(pred_test)) > 1:
            ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
            sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
            print("{}: CH={:.3f}, SC={:.5f}".format(link_method, ch_test, sc_test))
        else:
            print(f"{link_method}: Solo un cluster.")

    # DBSCAN
    print(f"\n----- Ejecutando DBSCAN (eps=0.2, min_samples=10) - {barrio} -----")
    dbscan = DBSCAN(eps=0.2, min_samples=10)
    t = time.time()
    dbscan_predict = dbscan.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    if len(np.unique(dbscan_predict)) > 1:
        evaluar_clustering(X_normal, dbscan_predict, "DBSCAN (eps=0.2, min_samples=10)", barrio)
        scatter_matrix(X, dbscan_predict, f"DBSCAN (eps=0.2,min_samples=10) - {barrio}", usadas, f"scatter_dbscan_{barrio}.pdf")
        violinplot_silhouette(X_normal, dbscan_predict, "DBSCAN (eps=0.2, min_samples=10)", barrio)
    else:
        print("DBSCAN no encontró clusters (todos son ruido). Ajustar parámetros.")

    print("\n--- Variando eps en DBSCAN (min_samples=10) ---")
    for eps_val in [0.1, 0.15, 0.2, 0.25, 0.35]:
        db_test = DBSCAN(eps=eps_val, min_samples=10)
        pred_test = db_test.fit_predict(X_normal)
        if len(np.unique(pred_test)) > 1:
            ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
            sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
            print("eps={}: Clusters={}, CH={:.3f}, SC={:.5f}".format(eps_val, len(np.unique(pred_test)), ch_test, sc_test))
        else:
            print(f"eps={eps_val}: todos ruido o un solo cluster.")

    # MeanShift
    print(f"\n----- Ejecutando MeanShift - {barrio} -----")
    t = time.time()
    meanshift = MeanShift(bandwidth=0.4)
    ms_predict = meanshift.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, ms_predict, "MeanShift", barrio)
    scatter_matrix(X, ms_predict, f"MeanShift - {barrio}", usadas, f"scatter_meanshift_{barrio}.pdf")

    print("Centroides MeanShift (bandwidth=0.4):")
    ms_centers = meanshift.cluster_centers_
    print(ms_centers)

    violinplot_silhouette(X_normal, ms_predict, "MeanShift", barrio)

    # Spectral Clustering
    print(f"\n----- Ejecutando SpectralClustering (n_clusters=4) - {barrio} -----")
    t = time.time()
    spectral = SpectralClustering(n_clusters=4, random_state=123)
    spectral_predict = spectral.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, spectral_predict, "SpectralClustering (n=4)", barrio)
    scatter_matrix(X, spectral_predict, f"SpectralClustering (n=4) - {barrio}", usadas, f"scatter_spectral_{barrio}.pdf")

    violinplot_silhouette(X_normal, spectral_predict, "SpectralClustering (n=4)", barrio)

    print(f"\nAnálisis completado para el barrio: {barrio}.")

##########################################
# Comparación visible entre Albaicín y Centro
##########################################
compare_barrios = ["Albaicín", "Centro"]
sc_kmeans_comparacion = {}

for barrio in compare_barrios:
    subdf = barrios[barrio]
    if subdf.empty:
        print(f"No hay datos para {barrio}, no se puede hacer comparación.")
        continue
    subdf = subdf.rename(columns={
        "Price avg": "precio", 
        "Distance": "distancia", 
        "Ranking position avg": "posicion",
        "Guests": "huespedes",
        "Surface Area (m2)": "superficie"
    })
    # Verificar columnas y no vacío
    if subdf.empty or not all(col in subdf.columns for col in usadas):
        print(f"No se puede comparar {barrio} porque faltan columnas o datos.")
        continue

    Xb = subdf[usadas].apply(norm_to_zero_one)
    if Xb.empty:
        print(f"{barrio} no tiene datos tras normalización. No se compara.")
        continue

    km4 = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=123456)
    pred = km4.fit_predict(Xb)
    if len(np.unique(pred)) > 1:
        sc = metrics.silhouette_score(Xb, pred, sample_size=floor(len(Xb)*0.5), random_state=123456)
        sc_kmeans_comparacion[barrio] = sc
    else:
        print(f"{barrio} con k=4 en K-Means da un solo cluster, no se puede comparar SC.")

if len(sc_kmeans_comparacion) == 2:
    plt.figure()
    sns.barplot(x=list(sc_kmeans_comparacion.keys()), y=list(sc_kmeans_comparacion.values()), palette='Paired')
    plt.title("Comparación SC de K-Means(k=4) entre Albaicín y Centro")
    plt.ylabel("Silhouette Coefficient")
    plt.xlabel("Barrio")
    plt.ylim(0,1)
    plt.savefig("comparacion_kmeans_albaicin_centro.pdf")
    plt.close()
    print("\nComparación visible completada: se ha generado 'comparacion_kmeans_albaicin_centro.pdf'.")
else:
    print("\nNo se pudo realizar la comparación visible de SC en K-Means(k=4) entre Albaicín y Centro.")
