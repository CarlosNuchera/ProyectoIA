# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:21:52 2021

@author: Sergio
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import networkx as nx
from sklearn import preprocessing, model_selection

gang = pandas.read_csv("data/gang_attr.csv", header=0, 
                       names=["Age","Birthplace","Residence","Arrests",
                              "Convictions","Prison","Music","Ranking"])
print(gang.head(5))

features = gang.loc[:, 'Age':'Prison']  # selección de las columnas de atributos
target = gang['Ranking']  # selección de la columna objetivo
codificador_features = preprocessing.OrdinalEncoder()
codificador_features.fit(features)
# Codificación de los atributos una vez ajustado el codificador
features_codificados = codificador_features.transform(features)

codificador_target = preprocessing.LabelEncoder()
# El método fit_transform ajusta la codificación y la aplica a los datos a continuación
target_codificado = codificador_target.fit_transform(target)

# Frecuencia total de cada clase de Ranking
print("\n", pandas.Series(target).value_counts(normalize=True))

# Divide el conjunto de datos en dos subconjuntos, uno de test y otro de entrenamiento
(features_train, features_test,
 target_train, target_test) = model_selection.train_test_split(
        # Conjuntos de datos a dividir, usando los mismos índices para ambos
        features_codificados, target_codificado,
        # Valor de la semilla aleatoria, para que el muestreo sea reproducible,
        # a pesar de ser aleatorio
        random_state=346582,
        # Tamaño del conjunto de prueba (porcentaje)
        test_size=1/3)

# Comprobamos que el conjunto de prueba contiene el 33 % de los datos, en la misma proporción
# con respecto a la variable objetivo
print("\nCantidad de ejemplos de pruebas requeridos:", 54 * 1/3)
print("Filas del array de atributos de prueba:", features_test.shape[0])
print("Longitud del vector de objetivos de prueba:", len(target_test))
print("Proporción de clases en el vector de objetivos de prueba:")
print(pandas.Series(
        codificador_target.inverse_transform(target_test)
      ).value_counts(normalize=True))
# Comprobamos que el conjunto de entrenamiento contiene el resto de los datos, en la misma
# proporción con respecto a la variable objetivo
print("\nCantidad de ejemplos de entrenamiento requeridos:", 54 * 2/3)
print("Filas del array de atributos de entrenamiento:", features_train.shape[0])
print("Longitud del vector de objetivos de entrenamiento:", len(target_train))
print("Proporción de clases en el vector de objetivos de entrenamiento:")
print(pandas.Series(
        codificador_target.inverse_transform(target_train)
      ).value_counts(normalize=True))































"""gd = pd.read_csv('data/gang.csv', index_col=0)
G = nx.DiGraph(gd.values)

print(nx.info(G))
plt.figure(figsize =(15, 15))
nx.draw_networkx(G, with_labels = True)"""

"""# DEGREE CENTRALITY
print("\nValores de Centralidad de grados:")
deg_centrality = nx.degree_centrality(G)
print(deg_centrality)

# CLOSENESS CENTRALITY
print("\nValores de Centralidad de cercanía:")
close_centrality = nx.closeness_centrality(G)
print(close_centrality)"""