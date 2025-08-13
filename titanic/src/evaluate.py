from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression # Import adicionado para o teste
from sklearn.ensemble import RandomForestClassifier # Import adicionado para o teste
from sklearn.datasets import make_classification # Import adicionado para o teste

import matplotlib.pyplot as plt
import numpy as np
import json

def evaluate_model_with_cross_val(model, X, y, scoring='accuracy', cv=5):
    print(f"\n\nExecutando validação cruzada para o modelo com métrica '{scoring}' e {cv} folds...")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1) # n_jobs=-1 usa todos os cores disponíveis
    print(f"Scores de validação cruzada: {scores}")
    print(f"Média dos scores de validação cruzada: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores


def evaluate_model(model, X_test, y_test, class_names=None):
    print("Avaliando o modelo no conjunto de teste...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1) # Precisão para a classe positiva (sobreviveu)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    print(f"Precisão no conjunto de teste: {precision:.4f}")
    print(f"Recall no conjunto de teste: {recall:.4f}")

    # Gerar a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_) # Garante a ordem das classes
    fig_cm, ax_cm = plt.subplots(figsize=(7, 7))
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names if class_names else model.classes_)
    disp_cm.plot(cmap=plt.cm.Blues, ax=ax_cm)
    ax_cm.set_title("Matriz de Confusão")
    plt.tight_layout()

    # Gerar o classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)
    print("\nRelatório de Classificação:\n", report)

    return metrics, fig_cm, report


if __name__ == '__main__':
    print("Executando teste do módulo evaluate.py com métricas expandidas para LR e RF...")

    # Gerar um dataset dummy para os testes
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=10, random_state=42, weights=[0.7, 0.3])
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy)

    class_names_dummy = ['Classe 0', 'Classe 1'] # Nomes para as classes dummy

    # --- Teste para Regressão Logística ---
    print("\n========== Testando Regressão Logística ==========")
    lr_model_dummy = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    lr_model_dummy.fit(X_train_dummy, y_train_dummy)

    print("\n--- Validação Cruzada (LR) ---")
    _ = evaluate_model_with_cross_val(lr_model_dummy, X_train_dummy, y_train_dummy, scoring='accuracy', cv=5)
    _ = evaluate_model_with_cross_val(lr_model_dummy, X_train_dummy, y_train_dummy, scoring='precision', cv=5)
    _ = evaluate_model_with_cross_val(lr_model_dummy, X_train_dummy, y_train_dummy, scoring='recall', cv=5)

    print("\n--- Avaliação Completa no Conjunto de Teste (LR) ---")
    lr_metrics_dummy, lr_cm_fig_dummy, lr_report_dummy = evaluate_model(lr_model_dummy, X_test_dummy, y_test_dummy, class_names=class_names_dummy)
    print(f"Métricas detalhadas do teste LR: {lr_metrics_dummy}")

    # Exemplo de salvamento para LR (descomente para testar o salvamento de arquivos)
    # save_metrics(lr_metrics_dummy, "outputs/test_lr_metrics.json")
    # save_confusion_matrix(lr_cm_fig_dummy, "outputs/test_lr_confusion_matrix.png")
    # save_classification_report(lr_report_dummy, "outputs/test_lr_classification_report.txt")


    # --- Teste para Floresta Aleatória ---
    print("\n========== Testando Floresta Aleatória ==========")
    rf_model_dummy = RandomForestClassifier(random_state=42)
    rf_model_dummy.fit(X_train_dummy, y_train_dummy)

    print("\n--- Validação Cruzada (RF) ---")
    _ = evaluate_model_with_cross_val(rf_model_dummy, X_train_dummy, y_train_dummy, scoring='accuracy', cv=5)
    _ = evaluate_model_with_cross_val(rf_model_dummy, X_train_dummy, y_train_dummy, scoring='precision', cv=5)
    _ = evaluate_model_with_cross_val(rf_model_dummy, X_train_dummy, y_train_dummy, scoring='recall', cv=5)

    print("\n--- Avaliação Completa no Conjunto de Teste (RF) ---")
    rf_metrics_dummy, rf_cm_fig_dummy, rf_report_dummy = evaluate_model(rf_model_dummy, X_test_dummy, y_test_dummy, class_names=class_names_dummy)
    print(f"Métricas detalhadas do teste RF: {rf_metrics_dummy}")

    # Exemplo de salvamento para RF (descomente para testar o salvamento de arquivos)
    # save_metrics(rf_metrics_dummy, "outputs/test_rf_metrics.json")
    # save_confusion_matrix(rf_cm_fig_dummy, "outputs/test_rf_confusion_matrix.png")
    # save_classification_report(rf_report_dummy, "outputs/test_rf_classification_report.txt")

    print("\nTestes de módulo evaluate.py concluídos.")