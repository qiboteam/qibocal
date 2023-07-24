from qibocal.fitting.classifier import run


# @pytest.fixture
def test_load_model(tmp_path):
    classifier = run.Classifier(run.import_classifiers(["qubit_fit"])[0], tmp_path)
    classifier.create_model({"par1": 1})
    classifier.dump_hyper(tmp_path)
    new_classifier = run.Classifier.model_from_dir(tmp_path / "qubit_fit")
    assert new_classifier == classifier.trainable_model
