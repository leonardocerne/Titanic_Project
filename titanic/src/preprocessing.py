import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Classe para preencher valores faltantes

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy = "median", column=None):
        self.strategy = strategy
        self.column = column
        self.fill_value = None

    def fit(self, X, y=None):
        if self.column == None:
            raise ValueError("Coluna precisa ser especificada.")

        if self.strategy == "median":
            self.fill_value = X[self.column].median()
        elif self.strategy == 'mode':
            self.fill_value = X[self.column].mode()[0]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'median' or 'mode'.")
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.column in X_copy.columns:
            X_copy[self.column] = X_copy[self.column].fillna(self.fill_value)
        return X_copy
 


# Classe para criar categoria age_cat

class AgeCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        bins = [0, 12, 19, 59, np.inf]
        labels = ['crianca', 'adolescente', 'adulto', 'idoso']
        X_copy['age_cat'] = pd.cut(X_copy['age'], bins=bins, labels=labels, right=True, include_lowest=True)
        return X_copy
    

# Criando Pipeline principal de pré processamento, incluindo nossos transformers customizados
def create_full_preprocessing_pipeline():
    numeric_features = ['age', 'fare']
    categorical_features = ['sex', 'embarked', 'pclass', 'age_cat', 'sibsp', 'parch']

    scaler_transformer = StandardScaler()
    onehot_transformer = OneHotEncoder(handle_unknown='ignore')

    final_preprocessing_step = ColumnTransformer(
        transformers=[
            ('num_scale', scaler_transformer, numeric_features),
            ('cat_onehot', onehot_transformer, categorical_features)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline(steps=[
        ('initial_transforms', Pipeline(steps=[
            ('age_imputer_pre', CustomImputer(strategy='median', column='age')),
            ('embarked_imputer_pre', CustomImputer(strategy='mode', column='embarked')),
            ('age_categorizer', AgeCategorizer())
        ])),
        ('final_processing', final_preprocessing_step)
    ])

    return full_pipeline


# --- Bloco de Teste/Exemplo (apenas para verificar o pipeline em si) ---
if __name__ == '__main__':
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    titanic_df = sns.load_dataset('titanic')
    X = titanic_df.drop('survived', axis=1)
    y = titanic_df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Testando o Pipeline de Pré-processamento ---")
    full_preprocessing_pipeline = create_full_preprocessing_pipeline()

    X_train_transformed = full_preprocessing_pipeline.fit_transform(X_train)

    print(f"\nFormato de X_train_transformed: {X_train_transformed.shape}")
    print("\nTipo do array transformado:", type(X_train_transformed))
    print("\nVerificando se há valores não numéricos no array transformado:")
    print(np.issubdtype(X_train_transformed.dtype, np.number))

    print("\nPrimeiras 5 linhas do array transformado:")
    print(X_train_transformed[:5])

    final_preprocessor_step = full_preprocessing_pipeline.named_steps['final_processing']
    transformed_feature_names = final_preprocessor_step.get_feature_names_out()
    print("\nNomes das features transformadas:")
    print(transformed_feature_names)

    print("\nTeste do Pipeline de Pré-processamento concluído.")