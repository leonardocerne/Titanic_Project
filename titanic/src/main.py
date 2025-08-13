import pandas as pd
from sklearn.model_selection import train_test_split
from src import load_data
from src import preprocessing
from src import train
from src import evaluate
from src import utils

def main():
    print('''___________________________________________________________________________
          \nIniciando projeto Titanic com ML
          \n___________________________________________________________________________''')
    data = load_data.load()

    # Separando X e y
    X = data.drop("survived", axis=1)
    y = data["survived"]

    print("...dividindo dados em set de treino e set de teste...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Criando pipeline completo de processamento  
    full_preprocessing_pipeline = preprocessing.create_full_preprocessing_pipeline()

    # Processando dados de treino e teste  
    X_treino_processado = full_preprocessing_pipeline.fit_transform(X_treino) # Fazemos fit somente no de treino para nao ter data leakage
    X_teste_processado = full_preprocessing_pipeline.transform(X_teste)

    # Dando nomes para survived ao invés de 0 ou 1 para melhor visualização  
    class_names = ["Não sobreviveu", "Sobreviveu"]

    # Usando regressão logística  
    print("\n--- Regressão Logística ---")
    lr_model = train.train_logistic_regression(X_treino_processado, y_treino)

    # Avaliando modelo  
    print("\nValidação Cruzada (Treino):")
    lr_cv_acc = evaluate.evaluate_model_with_cross_val(lr_model, X_treino_processado, y_treino, scoring="accuracy", cv=5)
    lr_cv_prec = evaluate.evaluate_model_with_cross_val(lr_model, X_treino_processado, y_treino, scoring="precision", cv=5)
    lr_cv_rec = evaluate.evaluate_model_with_cross_val(lr_model, X_treino_processado, y_treino, scoring='recall', cv=5)
    
    lr_metrics, lr_cm_fig, lr_report = evaluate.evaluate_model(lr_model, X_teste_processado, y_teste, class_names=class_names)

    # Salvando avaliações  
    utils.save_metrics(lr_metrics, "outputs/lr_test_metrics.json")
    utils.save_confusion_matrix(lr_cm_fig, "outputs/lr_confusion_matrix.png")
    utils.save_classification_report(lr_report, "outputs/lr_classification_report.txt")

    lr_y_pred_proba = lr_model.predict_proba(X_teste_processado)
    utils.plot_predictions(
        y_true=y_teste,
        y_pred=lr_model.predict(X_teste_processado), # Classes previstas (0 ou 1)
        y_prob=lr_y_pred_proba, # Probabilidades para plotar distribuição
        filepath="outputs/lr_predictions_distribution.png",
        title="Regressão Logística - Probabilidade de Sobrevivência"
    )

    # Usando floresta aleatória
    print("\n---Floresta Aleatória---")
    rf_model = train.train_random_forest(X_treino_processado, y_treino)

    print("\nValidação Cruzada (Treino):")
    rf_cv_acc = evaluate.evaluate_model_with_cross_val(rf_model, X_treino_processado, y_treino, scoring='accuracy', cv=5)
    rf_cv_prec = evaluate.evaluate_model_with_cross_val(rf_model, X_treino_processado, y_treino, scoring='precision', cv=5)
    rf_cv_rec = evaluate.evaluate_model_with_cross_val(rf_model, X_treino_processado, y_treino, scoring='recall', cv=5)

    # Avaliação completa no conjunto de teste
    rf_metrics, rf_cm_fig, rf_report = evaluate.evaluate_model(rf_model, X_teste_processado, y_teste, class_names=class_names)

    # Salvar resultados da Floresta Aleatória
    utils.save_metrics(rf_metrics, "outputs/rf_test_metrics.json")
    utils.save_confusion_matrix(rf_cm_fig, "outputs/rf_confusion_matrix.png")
    utils.save_classification_report(rf_report, "outputs/rf_classification_report.txt")  

    rf_y_pred_proba = rf_model.predict_proba(X_teste_processado)
    utils.plot_predictions(
        y_true=y_teste,
        y_pred=rf_model.predict(X_teste_processado),
        y_prob=rf_y_pred_proba,
        filepath="outputs/rf_predictions_distribution.png",
        title="Floresta Aleatória - Probabilidade de Sobrevivência"
    )

    print("\n--- Comparação Final de Modelos no Conjunto de Teste ---")
    print(f"Regressão Logística - Acurácia: {lr_metrics['accuracy']:.4f} | Precisão: {lr_metrics['precision']:.4f} | Recall: {lr_metrics['recall']:.4f}")
    print(f"Floresta Aleatória - Acurácia: {rf_metrics['accuracy']:.4f} | Precisão: {rf_metrics['precision']:.4f} | Recall: {rf_metrics['recall']:.4f}")

    # Exemplo de salvamento do "melhor" modelo (baseado na acurácia, por exemplo)
    if lr_metrics['accuracy'] > rf_metrics['accuracy']:
        utils.save_model(lr_model, "models/final_model.pkl")
        print("\nRegressão Logística selecionada como modelo final e salva.")
    else:
        utils.save_model(rf_model, "models/final_model.pkl")
        print("\nFloresta Aleatória selecionada como modelo final e salva.")

    print("\nProjeto Titanic ML concluído com sucesso!")  


if __name__ == '__main__':
    main()