from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scikit_normalize(constructor):
    r"""Returns a `Pipeline` with `StandardScaler` and the
    `constructor`.

    Args:

        constructor: `sklearn` model.
    """
    return make_pipeline(StandardScaler(), constructor)


def identity(x):
    return x
