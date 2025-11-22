
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

# Import custom modules
from src.data.preprocessor import TransactionPreprocessor
from src.data.augmentation import TransactionAugmenter
from src.models.baseline_models import BaselineClassifier
from src.evaluation.metrics import ModelEvaluator


class TrainingPipeline:
    """Training pipeline compatible with API"""

    def __init__(self):
        self.preprocessor = TransactionPreprocessor()
        self.augmenter = TransactionAugmenter(seed=42)
        print("✓ Training pipeline initialized")

    def load_data(self, data_path: str):
        print(f"\n📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        if "transaction_text" not in df.columns or "category" not in df.columns:
            raise ValueError("CSV must contain 'transaction_text' and 'category' columns")

        df = df.dropna(subset=["transaction_text", "category"])

        print(f"✓ Loaded {len(df)} rows with {df['category'].nunique()} categories.")
        return df

    def preprocess(self, df):
        print("\n🔄 Preprocessing text...")
        df["processed_text"] = self.preprocessor.preprocess_batch(
            df["transaction_text"].tolist()
        )
        print("✓ Preprocessing complete")
        return df

    def augment(self, df, multiplier=3):
        print(f"\n🔄 Augmenting x{multiplier} ...")
        return self.augmenter.augment_dataset(df, target_multiplier=multiplier)

    def split(self, df):
        print("\n📊 Splitting dataset (train/val/test)")

        train_df, test_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=df["category"]
        )

        train_df, val_df = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df["category"]
        )

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df

    def train(self, train_df):
        print("\n🚀 Training model...\n")

        X_train = train_df["processed_text"].tolist()
        y_train = train_df["category"].tolist()

        model = BaselineClassifier(model_type="logistic")
        model.train(X_train, y_train)
        return model

    def evaluate(self, model, test_df):
        print("\n📊 Evaluating model...\n")

        X_test = test_df["processed_text"].tolist()
        y_true = test_df["category"].tolist()

        preds = model.predict(X_test)
        y_pred = [p["predicted_category"] for p in preds]

        class_names = sorted(list(set(y_true)))

        evaluator = ModelEvaluator(class_names)
        results = evaluator.evaluate_text_labels(y_true, y_pred)
        evaluator.print_report(results)

        return results

    def save(self, model, output_dir="models/production"):
        print(f"\n💾 Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        model.save(f"{output_dir}/baseline_model.pkl")

        # save class names
        class_names = sorted(model.model.classes_)
        with open(f"{output_dir}/class_names.json", "w") as f:
            json_data = {"classes": class_names}
            import json
            json.dump(json_data, f, indent=2)

        print("✓ Model + class names saved successfully!")

    def run(self, csv_path):
        print("\n==============================")
        print("🎯 START TRAINING PIPELINE")
        print("==============================")

        df = self.load_data(csv_path)
        df = self.preprocess(df)

        df = self.augment(df, multiplier=3)
        df = self.preprocess(df)

        train_df, val_df, test_df = self.split(df)

        model = self.train(train_df)
        results = self.evaluate(model, test_df)

        self.save(model)

        print("\n==============================")
        print("🎉 TRAINING COMPLETE")
        print("==============================")

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    pipeline = TrainingPipeline()
    pipeline.run(args.data)
