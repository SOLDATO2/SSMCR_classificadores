import time
from flask import Flask, request, jsonify
import requests
import threading

app = Flask(__name__)

def enviar_dados():
    minha_rota = f'http://127.0.0.1:{5001}/receber_informacoes_do_classificador'  # Obtenha a porta atual do validador.py
    informacoes = []
    escolha = input("Gostaria de enviar um dado pre-definidos para enviar ao clustering? (yes/no): ")
    if escolha == "no":
        info = int(input("Qual é a idade do examinado?: "))
        informacoes.append(info)
        info =  input("Qual é o sexo do entrevistado?(Male/Female): ")
        informacoes.append(info)
        info = int(input("Quantos casos de sepsis o examinado teve? (ex: 1): "))
        informacoes.append(info)

    elif escolha == "yes":
        informacoes = [23, "Male", 1]
    
    
    print("Enviando dados...")
    
    try:
        response = requests.post('http://127.0.0.1:5000/receber_informacoes', json={'rota': minha_rota, 'informacoes': informacoes})
        if response.status_code == 200:
            print("Dados enviados com sucesso para o Classificador.")
        else:
            print("Falha ao enviar dados para o classificador. Tentando novamente...")
    except requests.exceptions.RequestException as e:
        print("Erro de conexão:", e)
    time.sleep(5)


@app.route('/receber_informacoes_do_classificador', methods=['POST'])
def receber_informacoes_do_classificador():
    
    if request.method == 'POST':
        dados = request.json
        informacoes_entrevistado = dados.get('informacoes_entrevistado')
        atributos_predict_forest_cross_com_acuracia = dados.get('atributos_predict_forest_cross_com_acuracia')
        atributos_predict_forest_cross_com_acuracia_PROBA = dados.get('atributos_predict_forest_cross_com_acuracia_PROBA')
    
    print("Instancia retornada do classificador: ", informacoes_entrevistado)
    print("Classificação atributos_predict_forest_cross_com_acuracia: ", atributos_predict_forest_cross_com_acuracia)
    print("Classificação atributos_predict_forest_cross_com_acuracia_PROBA: ", atributos_predict_forest_cross_com_acuracia_PROBA)
    print("SCORE ALIVE: ", atributos_predict_forest_cross_com_acuracia_PROBA[0])
    print("SCORE DEAD: ", atributos_predict_forest_cross_com_acuracia_PROBA[1])
    
    
    

    
    
    
    return "Operacao feita com sucesso"
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    thread_informacoes = threading.Thread(target=enviar_dados)
    thread_informacoes.start()
    
    app.run(host='127.0.0.1', port=5001)