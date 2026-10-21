import json
import os
import tempfile
import unittest
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
import app as student_app

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.tmp.name, "model.joblib")
        cls.meta_path = os.path.join(cls.tmp.name, "metadata.json")
        cls.features = ["studytime", "absences"]
        model = DummyRegressor(strategy="constant", constant=12)
        model.fit(pd.DataFrame([[2, 4], [3, 8]], columns=cls.features), [12, 12])
        joblib.dump(model, cls.model_path)
        metadata = {
            "features": cls.features,
            "input_schema": [
                {"name": "studytime", "type": "number", "min": 1, "max": 4, "step": 1},
                {"name": "absences", "type": "number", "min": 0, "max": 93, "step": 1}
            ],
            "model_name": "test_baseline"
        }
        with open(cls.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        student_app.MODEL_PATH = cls.model_path
        student_app.METADATA_PATH = cls.meta_path
        cls.client = student_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_ready(self):
        self.assertEqual(self.client.get("/ready").status_code, 200)

    def test_model_info(self):
        self.assertEqual(self.client.get("/model-info").status_code, 200)

    def test_valid_prediction(self):
        r = self.client.post("/predict", json={"studytime": 2, "absences": 4})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["predicted_grade"], 12.0)

    def test_missing_field(self):
        r = self.client.post("/predict", json={"studytime": 2})
        self.assertEqual(r.status_code, 400)

    def test_invalid_number(self):
        r = self.client.post("/predict", json={"studytime": "many", "absences": 4})
        self.assertEqual(r.status_code, 400)

    def test_out_of_range(self):
        r = self.client.post("/predict", json={"studytime": 8, "absences": 4})
        self.assertEqual(r.status_code, 400)

    def test_missing_artifacts(self):
        old_model, old_meta = student_app.MODEL_PATH, student_app.METADATA_PATH
        student_app.MODEL_PATH = os.path.join(self.tmp.name, "not-found.joblib")
        student_app.METADATA_PATH = os.path.join(self.tmp.name, "not-found.json")
        try:
            self.assertEqual(self.client.get("/ready").status_code, 503)
        finally:
            student_app.MODEL_PATH, student_app.METADATA_PATH = old_model, old_meta

if __name__ == "__main__":
    unittest.main()
