# Health Insurance

Propensão de clientes na modalidade Cross-Sell em uma seguradora de saúde

![Grafico](Img/health_insurance.JPG)

## 1.	Problema de negócio
Uma seguradora que oferece Seguro Saúde aos seus clientes, agora eles precisam da sua ajuda na construção de um modelo para prever se os segurados (clientes) do ano passado também terão interesse no Seguro Automóvel oferecido pela empresa.

Uma apólice de seguro é um acordo pelo qual uma empresa se compromete a fornecer uma garantia de compensação por perdas, danos, doenças ou morte em troca do pagamento de um prêmio específico. Um prêmio é uma quantia em dinheiro que o cliente precisa pagar regularmente a uma seguradora por esta garantia.

Por exemplo, você pode pagar um prêmio de Rs. 5.000 por ano para uma cobertura de seguro saúde de Rs. 200.000/- para que se, Deus me livre, você adoecer e precisar ser hospitalizado naquele ano, a seguradora arcará com o custo da hospitalização, etc., por até Rs. 200.000. Agora, se você está se perguntando como uma empresa pode arcar com custos de hospitalização tão altos quando cobra um prêmio de apenas Rs. 5000/-, é aí que o conceito de probabilidades entra em cena. Por exemplo, como você, pode haver 100 clientes que pagariam um prêmio de Rs. 5.000 todos os anos, mas apenas alguns deles (digamos 2-3) seriam hospitalizados naquele ano e nem todos. Dessa forma, todos compartilham o risco de todos os outros.

Assim como o seguro médico, existe o seguro automóvel, onde todos os anos o cliente precisa pagar um prêmio de determinado valor à seguradora para que, em caso de acidente com o veículo, a seguradora forneça uma indenização (chamada de 'soma assegurada') para o consumidor.

## 2.	Objetivo
Construir um modelo para prever se um cliente estaria interessado em seguro automóvel é extremamente útil para a empresa porque ela pode então planear a sua estratégia de comunicação para chegar a esses clientes e otimizar o seu modelo de negócio e receitas.

## 3.	Premissas do negócio
O custo unitário por cada contato do time comercial é de Rs 4
A receita bruta por cada cliente que aderisse ao seguro de veículo seria de Rs 40.

## 4.	Estratégia adotada para solução

Passo 1 - Descrição dos dados: O objetivo deste passo é ter um entendimento inicial de como os dados estão relacionados com o problema de negócio proposto, para tal, lançando mão de algumas métricas estatísticas de posição e distribuição.

Passo 2 - Engenharia de atributos: Neste passo foram criadas novas variáveis a partir das variáveis originais a fim de melhorar a qualidade dos dados facilitando seu o entendimento.

Passo 3 - Filtragem das variáveis: No processo de filtragem, busca-se selecionar e reter apenas as variáveis relevantes para a análise ou modelagem, com base nos objetivos do negócio e nas características do problema em questão.
 
Passo 4 - Análise exploratória: A análise exploratória dos dados (EDA) tem como principal objetivo proporcionar uma compreensão mais aprofundada dos dados, tais como: Compreender a distribuição das variáveis numéricas e categóricas, identificar outliers (valores atípicos), se existe desbalanceamento da variável resposta, correlações entre as variáveis entre outras análises.

Passo 5 - Preparação dos dados: Neste tópico, os dados serão transformados, ou seja, as variáveis categóricas serão convertidas em números, pois os algoritmos de machine learning não tem uma boa performance com dados não numéricos. Os dados também precisarão ser reescalados, ou seja, colocar as variáveis em uma escala comum.
 
Passo 6 - Seleção das variáveis: Neste passo o principal objetivo é selecionar as variáveis mais relevantes e descartar as menos importantes, para que estas sejam submetidas aos algoritmos de machine learning. Esta seleção é feita através de algoritmos específicos de seleção, com isso, busca-se reduzir a dimensionalidade dos dados sem perder em performance ao resultado final.

Passo 7 - Algoritmos de machine learning: Serão implementados alguns modelos de algoritmos de machine learning para possamos escolher o que melhor performou com os dados disponíveis.
 
Passo 8 - Ajuste dos hiperparâmetros: Com a definição do melhor modelo de machine learning, busca-se otimizar ainda mais a performance do modelo com alguns ajustes finos em seus parâmetros.

Passo 9 - Performance do negócio: Nesta etapa, o termo "performance do negócio" refere-se ao impacto que os modelos e análises têm nos resultados e metas de uma organização, pois busca garantir que as soluções propostas realmente tragam benefícios tangíveis para a empresa.

Passo 10 - Modelo em produção: Nesta etapa final, o melhor modelo treinado é implantado e usado em um ambiente operacional do mundo real. O objetivo principal deste passo de colocar um modelo em produção é transformar o trabalho de desenvolvimento de modelos em um sistema que pode fornecer valor contínuo e automático.

## 5.	Top 3 insights
Hipótese 1: Clientes que tem carteira de motorista teriam mais interesse em contratar um seguro de veículo?
Falsa: Clientes que tem carteira de motorista NÃO teriam mais interesse em contratar um seguro de veículo.

Hipótese 2: Clientes que já possuíam seguro anteriormente teriam mais interesse em contratar um novo seguro de veículo?
Falsa: Clientes que já possuíam seguro anteriormente NÃO teriam mais interesse em contratar um novo seguro de veículo.

Hipótese 3: Clientes que tem veículos mais novos teriam mais interesse em contratar um seguro de veículo?
Falsa: Clientes que tem veículos entre 1 e 2 anos teriam mais interesse em contratar um seguro de veículo.

## 6.	Aplicação do modelo de machine learning
Foram feitos testes com vários algoritmos de machine learning, utilizando a técnica de cross-validation e ajuste fino dos parâmetros.

## 7.	Performance do modelo de machine learning
O algoritmo de machine learning escolhido foi o ExtraTreesClassifier.

## 8.	Resultado do negócio
Conforme mostrado na figura abaixo, à medida que aumenta a quantidade de clientes consultados na base de clientes de seguro saúde, a propensão de interessados também aumenta, ao ponto que para alcançar cerca de 100% dos clientes interessados em adquirir um seguro de veículo é preciso ligar para somente quase 60% da base de clientes do seguro saúde.

 ![Grafico](Img/Grafico.JPG)

Do ponto de vista de negócio e assumindo que o custo unitário é de Rs 4 e a receita bruta é de Rs 40, segue na tabela abaixo qual o valor da receita em relação ao percentual de clientes consultados.  

![Tabela](Img/Tabela_algoritmos.JPG)

## 9.	Conclusão
As conclusões deste projeto de propensão de compra de seguro veicular revelaram que as variáveis disponíveis se mostraram insuficientes para impulsionar um desempenho mais robusto nos algoritmos de machine learning. Além disso, observou-se uma tendência significativa, indicando que motoristas sem carteira de motorista demonstraram maior interesse na aquisição de seguro de veículo em comparação com seus homólogos que possuem carteira de motorista. Essas descobertas destacam a importância de considerar variáveis adicionais e nuances comportamentais ao desenvolver modelos classificatórios para compreender as preferências dos consumidores em relação a seguro de veículo.

## 10.	Próximos passos
•	Fazer o balanceamento dos dados;
•	Testar outros algoritmos de machine learning;
•	Buscar novas variáveis para definir melhor o comportamento dos clientes.
