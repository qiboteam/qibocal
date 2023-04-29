from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scikit_normalize(constructor):
    r"""Returns a `Pipeline` with `StandardScaler` and the
    `constructor`.

    Args:

        constructor: `sklearn` model.
    """
    return make_pipeline(StandardScaler(), constructor)


def scikit_dump(model, path: Path):
    initial_type = [("float_input", FloatTensorType([1, 2]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(path.with_suffix(".onnx"), "wb") as f:
        f.write(onx.SerializeToString())


def identity(x):
    return x
