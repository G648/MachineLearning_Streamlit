import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics


#ETAPA 01 - CARREGAR DADOS
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        #carregamento dos dados
        if os.path.exists(caminho_arquivo):
            
            df = pd.read_csv(caminho_arquivo, encoding="latin1", sep=',')

            print("O arquivo foi carregado com sucesso!")

            return df
        else:
            print("O arquivo não foi encontrado dentro da pasta!")

            return None
    except Exception as e:
        print("Erro inesperado ao carregar o arquivo: ", e)

        return None

# --- chamar a função para armazenar o resultado --- 

dados = carregar_dados()


# ---------- Etapa 02 : PREPARAÇÃO E DIVISAO DOS DADOS ------------
# definição de X (features) e Y (Target)

if dados is not None:
    print(f"\nTotal de Registros carregados: {len(dados)}")
    print("Iniciando o pipeline de treinamento")

    TARGET_COLUMN = "Status_Final"

    #Etapa 2.1 - definição das features e target
    try:

        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]

        print(f"Features (X) definidas: {list(X.columns)}")
        print(f"Features (y) definidas: {TARGET_COLUMN}")

    except KeyError:
        
        print(f"\n ----- Erro Crítico -----")
        print(f"A coluna {TARGET_COLUMN} não foi encontrado no CSV")
        print(f"Colunas disponíveis: {list(dados.columns)}")
        print(f"Por favor, ajuste a variável 'TARGET_COLUMN' e tente novamente!")
        #se o target não for encontrado, irá encerrar o script!
        exit()

    #Etapa 2.2 - Divisão entre treino e teste
    print("\n ------ Dividindo dados em treino e teste... ------")
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=0.2,      # 20% dos dados serão utilizados para teste
        random_state= 42,   #Garantir a reprodutibilidade
        stratify=y          #Manter a proporção de aprovados e reprovados
    )

    print(f"Dados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")

    # Etapa 03: CRIAÇÃO DA PIPELINE DE ML

    print("\n------ Criando a pipeline de ML... ------")
    #scaler -> normalização dos dados (colocando tudo na mesma escala)
    #model -> aplica o modelo de regressão logística
    pipeline_model = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #ETAPA 04: TREINAMENTO E AVALIAÇÃO DOS DADOS/MODELO
    print("\n ------ Treinamento do modelo... ------")
    #treina a pipeline com os dados de treino
    pipeline_model.fit(X_train, y_train)

    print("modelo treinado. Avaliando com os dados de teste...")
    y_pred = pipeline_model.predict(X_test)

    #AVALIAÇÃO DE DESEMPENHO
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n ------ Relatório de avaliação geral ------")
    print(f"Acuráia Geral: {accuracy * 100:.2f}%")
    print("\nRelatório de classificação detalhado:")
    print(report)


    #ETAPA 05: SALVANDO O MODELO
    model_filename = 'modelo_previsao_desempenho.joblib'

    print(f"\nSalvando o pipeline treinado em.. {model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("Processo concluído com sucesso!")
    print(f"O arquivo '{model_filename}' está para ser utilizado!")

else:
    print("O pipeline não pode continuar pois os dados não forma carregados!")