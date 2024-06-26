#colunas ages: min max ok
#sex, trocar 0 por male e 1 para female ok
#episode number min max ok
#hospital_outcome: 0 para morto e 1 para vivo ok




#########################
import pandas as pd
dados = pd.read_csv('SSMCR_classificadores\\s41598-020-73558-3_sepsis_survival_primary_cohort.csv', sep=',')



dados.rename(columns={'sex_0male_1female': 'sex'}, inplace=True)
dados['sex'] = dados['sex'].map({0: 'Male', 1: 'Female'})

dados.rename(columns={'hospital_outcome_1alive_0dead': 'hospital_outcome'}, inplace=True)
dados['hospital_outcome'] = dados['hospital_outcome'].map({0: 'Dead', 1: 'Alive'}) 
#########################


dados_numericos = dados.drop(columns = ['sex', 'hospital_outcome'])
dados_categoricos = dados['sex']
dados_classificadores = dados['hospital_outcome']


from pickle import dump
from sklearn import preprocessing
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
dump(modelo_normalizador, open('SSMCR_classificadores\\modelo_normalizador.pkl', 'wb'))


dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, dtype=int)
#print(dados_categoricos_normalizados)
dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)


dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns = ['age_years', 'episode_number'])

#juntar com os dados categoricos normalizados
dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how = 'left')
#dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos, how = 'left')
dados_normalizados_final = dados_normalizados_final.join(dados_classificadores, how = 'left')



with open('SSMCR_classificadores\\sepsis_colunas_normalizadas.csv', 'w') as file: 
  colunas_para_salvar = dados_normalizados_final.columns[:-1]
  file.write(','.join(colunas_para_salvar))


print(dados_normalizados_final)

  
####################
####OVERSAMPLING####
####################
from imblearn.over_sampling import SMOTE

dados_atributos = dados_normalizados_final.drop(columns = ['hospital_outcome'])
#dados_classes = dados_normalizados_final.drop(columns = ['age_years', 'episode_number','Female','Male'])
dados_classes = dados_normalizados_final['hospital_outcome']


#Construir um objeto a partir do smote
resampler = SMOTE()

#Executar o balanceamento
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

from collections import Counter
classes_count = Counter(dados_classes_b)
classes_count_n = Counter(dados_classes)
print("Classes pre balanceamento: ", classes_count_n)
print("Classes pos balanceamento: ", classes_count)

dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)
dados_finais = dados_atributos_b.join(dados_classes_b, how ='left')
print(dados_finais)

###########################################################################################


import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
"""
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
"""


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}


pprint(random_grid)
rf_grid = GridSearchCV(forest, random_grid, refit=True, verbose=2)


rf_grid.fit(dados_atributos_b, dados_classes_b)

print(*rf_grid.best_params_)

max_depth_param = rf_grid.best_params_["max_depth"]
max_features = rf_grid.best_params_["max_features"]
n_estimators = rf_grid.best_params_["n_estimators"]



#Avaliação da acuracia com cross-validation
from pprint import pprint

#1. Quando a avaliação é realizada com cross_validation, dispensa-se o split da base
from sklearn.ensemble import RandomForestClassifier
#Construir um objeto para responder o indutor
forest = RandomForestClassifier(max_depth = max_depth_param, max_features = max_features, n_estimators = n_estimators) # Construir o objeto indutor

#2.Treinar o modelo
#2.1 Requisitos:
#    a) dados normalizados e balanceados
#    b) dados segmentados em atributos e classificados
sepsis_forest_cross_com_acuracia = forest.fit(dados_atributos_b, dados_classes_b)


#3. Avaliar a acuracia com cross-validation
from sklearn.model_selection import cross_validate
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(forest, dados_atributos_b, dados_classes_b, cv=10, scoring=scoring)
#print(scores_cross)
print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())



#############################################
##############SALVANDO FLORESTA##############
#############################################
from pickle import dump
dump(sepsis_forest_cross_com_acuracia, open('SSMCR_classificadores\\sepsis_forest_cross_com_acuracia.pkl', 'wb'))







