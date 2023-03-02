from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def scikit_normalize(constructor):
    return make_pipeline(StandardScaler(), constructor )
