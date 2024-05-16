#colunas ages: min max
#sex, trocar 0 por male e 1 para female
#episode number min max
#hospital_outcome: 0 para morto e 1 para vivo





import pandas as pd
dados = pd.read_csv('SSMCR_treinamento_classificadores\\s41598-020-73558-3_sepsis_survival_primary_cohort.csv', sep=',')

print(dados.head(5))

from imblearn.over_sampling import SMOTE

dados_atributos = dados.drop(columns = ['hospital_outcome_1alive_0dead'])
dados_classes = dados['hospital_outcome_1alive_0dead']

#Construir um objeto a partir do smote
resampler = SMOTE()

#Executar o balanceamento
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

from collections import Counter
classes_count = Counter(dados_classes_b)
classes_count_n = Counter(dados_classes)
print("Classes antes do balanceamento: ", classes_count_n)
print("Classes depois do balanceamento: ", classes_count)

dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)
dados_finais = dados_atributos_b.join(dados_classes_b, how ='left')
print(dados_finais)

###########################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dados_atributos_b = dados_finais.drop(columns=['hospital_outcome_1alive_0dead'])
dados_classe_b = dados_finais['hospital_outcome_1alive_0dead']

atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classe_b, test_size = 0.3)

forest = RandomForestClassifier()

#Treinar o modelo
fertility_forest = forest.fit(atributos_train, classes_train)

#Pretestar o modelo
Classe_test_predict = fertility_forest.predict(atributos_test)

#Comparar as classes inferidas no teste com as classes preservadas

i = 0
for i in range(0, len(classes_test)):
  print(classes_test.iloc[i][0], ' - ', Classe_test_predict[i])

  #Acuracia global do modelo
from sklearn import metrics
print('Acuracia global (provisoria):', metrics.accuracy_score(classes_test, Classe_test_predict))