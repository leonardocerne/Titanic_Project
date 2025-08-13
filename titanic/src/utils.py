import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

def save_model(model, filepath: str):
    try:
        joblib.dump(model, filepath)
        print(f"Modelo salvo com sucesso em: {filepath}")
    except Exception as e:
        print(f"Erro ao salvar o modelo em {filepath}: {e}")

def load_model(filepath:str):
    try:
        model = joblib.load(filepath)
        print(f"Modelo carregado com sucesso de: {filepath}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo de {filepath}: {e}")
        return None
    
def save_metrics(metrics: dict, filepath: str):
    #Salva as métricas em um arquivo JSON
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas salvas em {filepath}")

def save_confusion_matrix(fig: plt.Figure, filepath: str):
    #Salva a figura da matriz de confusão
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Matriz de confusão salva em {filepath}")

def save_classification_report(report_string: str, filepath: str):
    #Salva o classification report em um arquivo de texto
    with open(filepath, 'w') as f:
        f.write(report_string)
    print(f"Relatório de classificação salvo em {filepath}")


def plot_predictions(y_true, y_pred, y_prob=None, filepath: str = None, title: str = "Previsões do Modelo"):
    fig, ax = plt.subplots(figsize=(8, 6))

    if y_prob is not None and y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_prob_positive = y_prob[:, 1]
    elif y_prob is not None:
        y_prob_positive = y_prob
    else:
        y_prob_positive = None

    if y_prob_positive is not None:
        ax.hist(y_prob_positive[y_true == 0], bins=20, alpha=0.7, label='Classe Real 0 (Não Sobreviveu)', color='red')
        ax.hist(y_prob_positive[y_true == 1], bins=20, alpha=0.7, label='Classe Real 1 (Sobreviveu)', color='green')
        ax.set_title(f"{title} - Distribuição das Probabilidades Previstas")
        ax.set_xlabel("Probabilidade Prevista de Sobreviver")
        ax.set_ylabel("Frequência")
        ax.legend()
    else:
        jitter = np.random.rand(len(y_true)) * 0.1 - 0.05
        ax.scatter(y_true + jitter, y_pred + jitter, alpha=0.6)
        ax.set_title(f"{title} - Previsões vs. Real")
        ax.set_xlabel("Valor Real (y_true)")
        ax.set_ylabel("Valor Previsto (y_pred)")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlim([-0.2, 1.2])
        ax.set_ylim([-0.2, 1.2])

    if filepath:
        try:
            fig.savefig(filepath)
            plt.close(fig)
            print(f"Gráfico de previsões salvo em: {filepath}")
        except Exception as e:
            print(f"Erro ao salvar gráfico de previsões em {filepath}: {e}")
    else:
        plt.show()