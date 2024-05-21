from pickle import load
from flask import Flask, request
import pandas as pd
import requests



app = Flask(__name__)



@app.route('/receber_informacoes', methods=['GET','POST'])
def receber_informacoes():

    if request.method == 'POST':
        dados = request.json
        rota = dados.get('rota')
        nova_instancia = dados.get('informacoes')



    nova_instancia_dict = {
            'age_years': nova_instancia[0],
            'sex_0male_1female': nova_instancia[1],
            'episode_number': nova_instancia[2],
            #'hospital_outcome_1alive_0dead': nova_instancia[3]
        }

    # Convertendo para DataFrame
    nova_instancia_df = pd.DataFrame([nova_instancia_dict])


    nova_instancia_df.rename(columns={'sex_0male_1female': 'sex'}, inplace=True)


    #nova_instancia_df.rename(columns={'hospital_outcome_1alive_0dead': 'hospital_outcome'}, inplace=True)
    #nova_instancia_df['hospital_outcome'] = nova_instancia_df['hospital_outcome'].map({0: 'Dead', 1: 'Alive'}) 
    #########################


    #dados_numericos = nova_instancia_df.drop(columns = ['sex', 'hospital_outcome'])
    dados_numericos = nova_instancia_df.drop(columns = ['sex'])
    dados_categoricos = nova_instancia_df['sex']
    #dados_classificadores = nova_instancia_df['hospital_outcome']



    #Normalizado min max
    normalizador = load(open('SSMCR_classificadores\\modelo_normalizador.pkl', 'rb'))
    dados_numericos_normalizados = normalizador.fit_transform(dados_numericos)

    dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns = ['age_years', 'episode_number'])

    #juntar com os dados categoricos normalizados
    dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, dtype=int)
    dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how = 'left')
    #dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos, how = 'left')
    #dados_normalizados_final = dados_normalizados_final.join(dados_classificadores, how = 'left')

    print(dados_normalizados_final)









    with open('SSMCR_classificadores\\Classificadores\\SSMCR_treinamento_classificadores\\sepsis_colunas_normalizadas.csv', 'r') as file:
        # Ler a primeira linha do arquivo, que contém os nomes das colunas separados por vírgulas
        columns = file.readline().strip().split(',')

    # Criar DataFrame vazio com as colunas obtidas
    data_frame = pd.DataFrame(columns=columns)


    nova_instancia_final_normalizada_ORGANIZADA_df = data_frame.copy()
    nova_instancia_final_normalizada_ORGANIZADA_df[dados_normalizados_final.columns] = dados_normalizados_final
    # Preencher as colunas faltantes com zeros
    print(nova_instancia_final_normalizada_ORGANIZADA_df)
    nova_instancia_final_normalizada_ORGANIZADA_df = nova_instancia_final_normalizada_ORGANIZADA_df.fillna(0)
    print(nova_instancia_final_normalizada_ORGANIZADA_df)





    forest = load(open('SSMCR_classificadores\\sepsis_forest_class.pkl', 'rb'))
    atributos_predict_forest_class = forest.predict(nova_instancia_final_normalizada_ORGANIZADA_df) # ['Alive']
    atributos_predict_forest_class_PROBA = forest.predict_proba(nova_instancia_final_normalizada_ORGANIZADA_df) # [[0.91400516 0.08599484]] ?
    
    forest_cross = load(open('SSMCR_classificadores\\sepsis_forest_cross.pkl', 'rb'))
    atributos_predict_forest_cross = forest_cross.predict(nova_instancia_final_normalizada_ORGANIZADA_df) #['Alive']
    atributos_predict_forest_cross_PROBA = forest_cross.predict_proba(nova_instancia_final_normalizada_ORGANIZADA_df) # [[0.91148461 0.08851539]]
    
    forest_cross_com_acuracia = load(open('SSMCR_classificadores\\sepsis_forest_cross_com_acuracia.pkl', 'rb'))
    atributos_predict_forest_cross_com_acuracia = forest_cross_com_acuracia.predict(nova_instancia_final_normalizada_ORGANIZADA_df) # ['Alive']
    atributos_predict_forest_cross_com_acuracia_PROBA = forest_cross_com_acuracia.predict_proba(nova_instancia_final_normalizada_ORGANIZADA_df) # [[0.91110789 0.08889211]]


    print(atributos_predict_forest_cross_com_acuracia)
    print(atributos_predict_forest_cross_com_acuracia_PROBA)
    
    
    #Forma Alternativa de printar proba & indicie 
    #print("##########################################################################")
    #dist_proba = forest_cross.predict_proba(nova_instancia_final_normalizada_ORGANIZADA_df)
    #import numpy as np
    #indice = np.argmax(dist_proba[0])
    #classe_predita = forest_cross.classes_[indice]
    #score = dist_proba[0][indice]
    #print("##########################################################################")
    #print("Classificado como: ", classe_predita, "Score: ", str(score))
    #print(np.argmax(dist_proba[0]))
    #print(forest_cross.classes_)
    #print("##########################################################################")



    colunas_categoricas_desnormalizadas = pd.from_dummies(nova_instancia_final_normalizada_ORGANIZADA_df[['Female', 'Male']])
    colunas_numericas_desnormalizadas = normalizador.inverse_transform(nova_instancia_final_normalizada_ORGANIZADA_df[['age_years', 'episode_number']])

    
    

    idade = colunas_numericas_desnormalizadas[0][0]
    casos_sepsis = colunas_numericas_desnormalizadas[0][1]
    sexo = colunas_categoricas_desnormalizadas.iloc[0][0]
 
    

    informacoes_entrevistado = {
        "idade": idade,
        "casos_sepsis": casos_sepsis,
        "sexo": sexo
    }




    





    dados = {
        "informacoes_entrevistado": informacoes_entrevistado,
        "atributos_predict_forest_class": atributos_predict_forest_class[0],
        "atributos_predict_forest_class_PROBA": atributos_predict_forest_class_PROBA[0].tolist(),
        "atributos_predict_forest_cross" : atributos_predict_forest_cross[0],
        "atributos_predict_forest_cross_PROBA": atributos_predict_forest_cross_PROBA[0].tolist(),
        "atributos_predict_forest_cross_com_acuracia": atributos_predict_forest_cross_com_acuracia[0],
        "atributos_predict_forest_cross_com_acuracia_PROBA": atributos_predict_forest_cross_com_acuracia_PROBA[0].tolist()
        
    }
    
    print(dados)
    
    # Enviar solicitação POST
    response = requests.post(rota, json=dados)

    # Verificar o status da resposta
    if response.status_code == 200:
        print("Informações enviadas com sucesso para a rota.")
    else:
        print("Ocorreu um erro ao enviar informações para a rota.")
    return "Operacao feita"
    
    
    
    


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)



