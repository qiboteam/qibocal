from pathlib import Path

import numpy as np
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.convert import to_onnx
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scikit_predict(loading_path: Path, input: np.typing.NDArray):
    r"""This function loads the scikit model saved in `loading_path`
    and returns the predictions of `input`.
    """
    sess = rt.InferenceSession(loading_path)
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: input.astype(np.float32)})[0]


def scikit_normalize(constructor):
    r"""Returns a `Pipeline` with `StandardScaler` and the
    `constructor`.

    Args:

        constructor: `sklearn` model.
    """
    return make_pipeline(StandardScaler(), constructor)


def scikit_dump(model, path: Path):
    r"""Dumps scikit `model` in `path`"""
    initial_type = [("float_input", FloatTensorType([None, 2]))]
    onx = to_onnx(model, initial_types=initial_type)
    with open(path.with_suffix(".onnx"), "wb") as f:
        f.write(onx.SerializeToString())
