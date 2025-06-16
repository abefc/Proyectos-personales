# -*- coding: utf-8 -*-
"""
Caso de estudio: Apartamentos con Precio Promedio < 125

Se aplican los mismos algoritmos de clustering, mismas visualizaciones y análisis que en los otros.
Además, se añade un preprocesamiento para tratar valores nulos en 'Type':
 - Si 'Type' está vacío, se mira la primera palabra de 'Description'.
   * 'Apartamento', 'Casa', 'Estudio' => Type = 'Apartment'
   * 'Hotel', 'Suite', 'Habitación' => Type = 'Hotel'
   * De lo contrario, Type = tipo más frecuente en todo el dataset.
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

def evaluar_clustering(X_normal, labels, nombre_alg, caso):
    if len(np.unique(labels)) < 2:
        print(f"\n--- Resultados {nombre_alg} - {caso} ---")
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
    print(f"\n--- Resultados {nombre_alg} - {caso} ---")
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

def violinplot_silhouette(X_normal, labels, algoritmo, caso):
    if len(np.unique(labels)) < 2:
        return
    sil_values = metrics.silhouette_samples(X_normal, labels)
    df_sil = pd.DataFrame({'cluster': labels, 'silhouette': sil_values})
    plt.figure()
    sns.violinplot(x='cluster', y='silhouette', data=df_sil, inner='box', palette='Paired')
    plt.title(f"Distribución del coeficiente de silhouette por cluster - {algoritmo} - {caso}")
    plt.savefig(f"violinplot_{algoritmo.replace(' ', '_')}_{caso}.pdf")
    plt.close()

###########################
# Carga de datos y preprocesamiento
###########################
datos = pd.read_csv('alojamientos_booking_Granada_2024.csv', sep=';', encoding="iso-8859-1")

# Imputación de superficie
media_superficie = datos.loc[datos["Surface Area (m2)"] > 0, "Surface Area (m2)"].mean()
datos.loc[datos["Surface Area (m2)"] == 0, "Surface Area (m2)"] = media_superficie

# Tratamiento de 'Type' nulo:
# 1. Encontrar el tipo más frecuente
tipo_mas_frecuente = datos['Type'].dropna().mode()
if len(tipo_mas_frecuente) > 0:
    tipo_mas_frecuente = tipo_mas_frecuente[0]
else:
    # Si no hay ninguno, por seguridad, establecemos un tipo por defecto
    tipo_mas_frecuente = "Apartment"

# 2. Rellenar nulos o vacíos
def imputar_type(row):
    current_type = row['Type']
    desc = row['Description']
    if not pd.isna(current_type) and current_type.strip() != '':
        # Ya tiene un tipo
        return current_type.strip().lower()
    else:
        # Sin tipo, miramos la primera palabra de Description
        if pd.isna(desc) or desc.strip() == '':
            # Sin descripción, usar el tipo más frecuente
            return tipo_mas_frecuente.lower()
        primera_palabra = desc.strip().split()[0].capitalize()  # Capitalizar para uniformidad
        if primera_palabra in ['Apartamento', 'Casa', 'Estudio']:
            return 'apartamento'
        elif primera_palabra in ['Hotel', 'Suite', 'Habitación']:
            return 'hotel'
        else:
            return tipo_mas_frecuente.lower()

datos['Type'] = datos.apply(imputar_type, axis=1)
datos.to_csv("comprobar.csv", sep=';', encoding="iso-8859-1")

# Definición del caso de estudio: Apartamentos con Precio Promedio < 125
casos = {
    "Apartamento": datos[
        (datos['Type'].str.contains('apartamento', case=False, na=False)) &
        (datos['Rating'] > 0) &
        (datos['Price avg'] < 350)
    ]
}

usadas = ['precio', 'distancia', 'posicion', 'huespedes', 'superficie']
resultados_kmeans = {}

for caso, subset in casos.items():
    print(f"\n\n********** Caso de estudio: {caso} **********")

    subset = subset.rename(columns={
        "Price avg": "precio",
        "Distance": "distancia",
        "Ranking position avg": "posicion",
        "Guests": "huespedes",
        "Surface Area (m2)": "superficie"
    })

    # Comprobar que el subset tiene datos
    if subset.empty:
        print(f"No hay datos para {caso}. Se omite este caso.")
        continue

    # Comprobar que las columnas existen
    columnas_faltantes = [col for col in usadas if col not in subset.columns]
    if columnas_faltantes:
        print(f"Las columnas {columnas_faltantes} no existen en {caso} tras renombrar. Se omite este caso.")
        continue

    X = subset[usadas]
    if X.empty:
        print(f"{caso} no tiene datos tras selección de variables.")
        continue

    X_normal = X.apply(norm_to_zero_one)

    ###########################
    # K-Means con k=4
    ###########################
    print("----- Ejecutando K-Means con k=4 -----")
    t = time.time()
    k_means = KMeans(init='k-means++', n_clusters=4, n_init=5, random_state=123456)
    cluster_predict = k_means.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    clusters_kmeans, ch_kmeans, sc_kmeans = evaluar_clustering(X_normal, cluster_predict, "K-Means (k=4)", caso)

    if sc_kmeans is not None:
        resultados_kmeans[caso] = sc_kmeans

    centers = pd.DataFrame(k_means.cluster_centers_, columns=list(X))
    centers_desnormal = centers.copy()
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    plt.figure()
    centers.index += 1
    hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
    hm.set_ylim(len(centers), 0)
    hm.figure.set_size_inches(10,10)
    hm.figure.savefig(f"centroides_kmeans_{caso}.pdf")
    centers.index -= 1
    plt.close()

    scatter_matrix(X, cluster_predict, f"K-Means (k=4) - {caso}", usadas, f"scatter_kmeans_{caso}.pdf")

    dist_centers = metrics.pairwise_distances(k_means.cluster_centers_)
    mds = MDS(n_components=2, random_state=123)
    coords = mds.fit_transform(dist_centers)
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], s=300, c=range(len(coords)), cmap='Paired')
    for i, (x_m, y_m) in enumerate(coords):
        plt.text(x_m, y_m, "C{}".format(i+1), fontsize=14, ha='center', va='center', color='black')
    plt.title(f"MDS sobre centroides de K-Means - {caso}")
    plt.savefig(f"mds_centroides_kmeans_{caso}.pdf")
    plt.close()

    violinplot_silhouette(X_normal, cluster_predict, "K-Means (k=4)", caso)

    print("\n--- Variando k en K-Means ---")
    for k_test in [2,3,4,5,6]:
        km_test = KMeans(n_clusters=k_test, n_init=5, random_state=123456)
        pred_test = km_test.fit_predict(X_normal)
        if len(np.unique(pred_test)) > 1:
            ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
            sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
            print("k={}: CH={:.3f}, SC={:.5f}".format(k_test, ch_test, sc_test))
        else:
            print(f"k={k_test}: Solo un cluster, no SC ni CH.")

    # Clustering Jerárquico
    print(f"\n----- Ejecutando Clustering Jerárquico (linkage='ward', n_clusters=4) - {caso} -----")
    hier = AgglomerativeClustering(n_clusters=4, linkage='ward')
    t = time.time()
    hier_predict = hier.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, hier_predict, "Jerárquico (ward, n=4)", caso)

    Z = sch.linkage(X_normal, 'ward')
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z, truncate_mode='level', p=10)
    plt.title(f"Dendrograma (Ward) - {caso}")
    plt.savefig(f"dendrograma_ward_{caso}.pdf")
    plt.close()

    scatter_matrix(X, hier_predict, f"Jerárquico (ward) - {caso}", usadas, f"scatter_hier_ward_{caso}.pdf")
    violinplot_silhouette(X_normal, hier_predict, "Jerárquico (ward, n=4)", caso)

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
    print(f"\n----- Ejecutando DBSCAN (eps=0.2, min_samples=10) - {caso} -----")
    dbscan = DBSCAN(eps=0.2, min_samples=10)
    t = time.time()
    dbscan_predict = dbscan.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    if len(np.unique(dbscan_predict)) > 1:
        evaluar_clustering(X_normal, dbscan_predict, "DBSCAN (eps=0.2, min_samples=10)", caso)
        scatter_matrix(X, dbscan_predict, f"DBSCAN (eps=0.2,min_samples=10) - {caso}", usadas, f"scatter_dbscan_{caso}.pdf")
        violinplot_silhouette(X_normal, dbscan_predict, "DBSCAN (eps=0.2, min_samples=10)", caso)
    else:
        print("DBSCAN no encontró clusters (todos son ruido). Ajustar parámetros.")

    print("\n--- Variando eps en DBSCAN (min_samples=10) ---")
    for eps_val in [0.1, 0.15, 0.2, 0.25, 0.35]:
        db_test = DBSCAN(eps=eps_val, min_samples=10)
        pred_test = db_test.fit_predict(X_normal)
        if len(np.unique(pred_test)) > 1:
            ch_test = metrics.calinski_harabasz_score(X_normal, pred_test)
            # Comprobar clusters antes de Silhouette
            if len(np.unique(pred_test)) > 1:
                sc_test = metrics.silhouette_score(X_normal, pred_test, sample_size=floor(len(X_normal)*0.5), random_state=123456)
                print("eps={}: Clusters={}, CH={:.3f}, SC={:.5f}".format(eps_val, len(np.unique(pred_test)), ch_test, sc_test))
            else:
                print(f"eps={eps_val}: Solo un cluster, no SC.")
        else:
            print(f"eps={eps_val}: todos ruido o un solo cluster.")

    # MeanShift
    print(f"\n----- Ejecutando MeanShift - {caso} -----")
    t = time.time()
    meanshift = MeanShift()
    ms_predict = meanshift.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, ms_predict, "MeanShift", caso)
    scatter_matrix(X, ms_predict, f"MeanShift - {caso}", usadas, f"scatter_meanshift_{caso}.pdf")

    print("Centroides MeanShift (bandwidth automático):")
    ms_centers = meanshift.cluster_centers_
    print(ms_centers)

    violinplot_silhouette(X_normal, ms_predict, "MeanShift", caso)

    # Spectral Clustering
    print(f"\n----- Ejecutando SpectralClustering (n_clusters=4) - {caso} -----")
    t = time.time()
    spectral = SpectralClustering(n_clusters=4, random_state=123)
    spectral_predict = spectral.fit_predict(X_normal)
    tiempo = time.time() - t
    print("Tiempo: {:.2f}s".format(tiempo))

    evaluar_clustering(X_normal, spectral_predict, "SpectralClustering (n=4)", caso)
    scatter_matrix(X, spectral_predict, f"SpectralClustering (n=4) - {caso}", usadas, f"scatter_spectral_{caso}.pdf")

    violinplot_silhouette(X_normal, spectral_predict, "SpectralClustering (n=4)", caso)

    print(f"\nAnálisis completado para el caso de estudio: {caso}.")

##########################################
# Eliminación de la Comparación entre Tipos
##########################################
# Dado que ahora solo hay un caso de estudio, se elimina la sección de comparación.
# No se generará el gráfico 'comparacion_kmeans_silhouette_hotel_vs_apartamento.pdf'.
# Si deseas mantener el código por futuras modificaciones, puedes comentar o eliminar esta sección.

print("\nAnálisis completado para el caso de estudio único: Apartamento con Precio Promedio < 125.")
