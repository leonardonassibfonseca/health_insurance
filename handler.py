import pickle
import pandas as pd
from flask import Flask, request, Response
# Importando classe HealthInsurance do arquivo HealthInsurance.py da pasta rossmann
# from nome_pasta ponto nome_arquivo import nome da classe dentro do arquivo

# Pacote é a pasta onde está o script.py
# Nome do arquivo.py
# Nome da classe que foi criada dentro do arquivo.py
from codigo.healthinsurance import HealthInsurance

# Colocando o caminho do arquivo do modelo treinado que foi salvo no Pickle
path = ''
modelo = pickle.load(open(path + 'modelo/PA004.pkl', 'rb'))

# Instaciando objeto da classe Flask que será a API
app = Flask(__name__)

# Método POST envia alguma coisa
# Método GET pede alguma coisa
# URL barra predict | Neste EndPoint/predict | EndPoint é tudo que vem depois da url principal, por exemplo: Globo.com/.....


@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    teste_json = request.get_json()  # Recebe um arquivo JSON a partir da request

    if teste_json:  # Se o teste_json for diferente de vazio, ou seja, se foi carregado algum dado
        # Verifica se o arquivo passado é um tipo de dicionário e se sim, foi enviado um arquivo com somente uma linha
        if isinstance(teste_json, dict):
            # Cria um dataframe e para isso é necessário indicar no Pandas qual é o nº da linha inicial, nesta caso, 0
            dados_que_vieram_da_producao = pd.DataFrame(teste_json, index=[0])
        else:
            # Se não for é um dicionário, foi enviado um arquivo com mais de uma linha
            dados_que_vieram_da_producao = pd.DataFrame(
                teste_json, columns=teste_json[0].keys())

        # Instanciando a classe do projeto, neste caso, HealthInsurance
        pipeline = HealthInsurance()

        # engenharia_de_atributos
        df5 = pipeline.engenharia_de_atributos(dados_que_vieram_da_producao)

        # modelagem_dos_dados
        df6 = pipeline.modelagem_dos_dados(df5)

        # predição
        df_resposta = pipeline.get_prediction(
            modelo, dados_que_vieram_da_producao, df6)

        return df_resposta
    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    # Dizer para endpoint rodar no localhost (rodando na máquina)
    app.run('0.0.0.0')
# 172.25.114.131 -> endereço IPv4 pc local
# app.run('0.0.0.0')
