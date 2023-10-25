import numpy as np

from qibocal.fitting.classifier import run

MODEL_FILE = "model.skops"
"""Filename for storing the model."""


def test_load_model(tmp_path):
    classifier = run.Classifier(run.import_classifiers(["qubit_fit"])[0], tmp_path)
    classifier.create_model({"par1": 1})
    classifier.dump_hyper(tmp_path)
    new_classifier = run.Classifier.model_from_dir(tmp_path / "qubit_fit")
    assert new_classifier == classifier.trainable_model


def test_predict_from_file(tmp_path):
    """Testing predict_from_file method."""
    classifier = run.Classifier(run.import_classifiers(["qubit_fit"])[0], tmp_path)
    model = classifier.create_model({"par1": 1})
    iqs = np.random.rand(10, 2)
    classifier.mod.dump(model, classifier.base_dir / MODEL_FILE)
    target_predictions = model.predict(iqs)
    predictions = classifier.mod.predict_from_file(tmp_path / MODEL_FILE, iqs)
    assert np.array_equal(target_predictions, predictions)
