import pickle
import pandas as pd
import numpy as np


class HealthInsurance:
    def __init__(self):
        # Colocando os caminhos dos arquivos que foram transformados e salvos no Pickle
        self.home_path = '/home/leonardo/projetos_/propensao_de_compra/projeto_completo/'
        self.Annual_Premium_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Annual_Premium_scaler.pkl', 'rb'))
        self.Age_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Age_scaler.pkl', 'rb'))
        self.Vintage_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Vintage_scaler.pkl', 'rb'))
        self.Gender_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Gender_scaler.pkl', 'rb'))
        self.Region_Code_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Region_Code_scaler.pkl', 'rb'))
        self.Policy_Sales_Channel_scaler = pickle.load(
            open(self.home_path + 'transformacoes/Policy_Sales_Channel_scaler.pkl', 'rb'))

    def engenharia_de_atributos(self, df1):
        # Alterando os dados da coluna 'Vehicle_Age' e 'Vehicle_Damage'
        df1['Vehicle_Age'] = df1['Vehicle_Age'].apply(
            lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year')
        df1['Vehicle_Damage'] = df1['Vehicle_Damage'].apply(
            lambda x: 1 if x == 'Yes' else 0)
        return df1

    def modelagem_dos_dados(self, df5):
        # Annual_Premium
        df5['Annual_Premium'] = self.Annual_Premium_scaler.transform(
            df5[['Annual_Premium']].values)  # Precisa passar um array

        # Age
        df5['Age'] = self.Age_scaler.transform(
            df5[['Age']].values)  # Precisa passar um array

        # Vintage
        df5['Vintage'] = self.Vintage_scaler.transform(
            df5[['Vintage']].values)  # Precisa passar um array

        # Gender -> Target Encoding
        # Target Encoding
        df5.loc[:, 'Gender'] = df5['Gender'].map(self.Gender_scaler)

        # Region_Code -> Target Encoding
        # Target Encoding
        df5.loc[:, 'Region_Code'] = df5['Region_Code'].map(
            self.Region_Code_scaler)

        # Vehicle_Age -> One Hot Encoding ou Ordinal Encoding
        # One Hot Encoding
        df5 = pd.get_dummies(df5, prefix='Vehicle_Age',
                             columns=['Vehicle_Age'])

        # Policy_Sales_Channel -> Target Encoding ou Frequency Encoding
        # Frequency Encoding
        df5.loc[:, 'Policy_Sales_Channel'] = df5['Policy_Sales_Channel'].map(
            self.Policy_Sales_Channel_scaler)

        # Colunas selecionadas a partir a importância das variáveis
        colunas_selecionadas = ['Annual_Premium',
                                'Vintage',
                                'Age',
                                'Region_Code',
                                'Vehicle_Damage',
                                'Policy_Sales_Channel',
                                'Previously_Insured']
        return df5[colunas_selecionadas]

    def get_prediction(self, modelo, dados_original, dados_teste):
        # Predição
        pred = modelo.predict_proba(dados_teste)

        # Juntando a predição com os dados originais
        dados_original['Score'] = pred[:, 1].tolist()

        return dados_original.to_json(orient='records', date_format='iso')
