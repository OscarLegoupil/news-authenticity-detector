#!/usr/bin/env python3
"""
Demo script showing how to use the fake news detection pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import FakeNewsDetector
from src.models.traditional.bow_tfidf import TFIDFClassifier
from src.models.transformers.deberta_classifier import TransformerClassifier as DeBERTaClassifier, ModelConfig
from src.data.preprocessing import TextPreprocessor, DatasetLoader

def demo_basic_usage():
    """Demonstrate basic usage of individual components."""
    print("=== Demo: Basic Component Usage ===\n")
    
    # 1. Text Preprocessing
    print("1. Text Preprocessing Demo")
    preprocessor = TextPreprocessor()
    
    sample_text = "Breaking: Scientists at MIT have discovered AMAZING new technology! Check out this INCREDIBLE breakthrough that will change everything!!! http://fake-news-site.com"
    
    print(f"Original: {sample_text}")
    print(f"Traditional: {preprocessor.clean_text_traditional(sample_text)}")
    print(f"Transformer: {preprocessor.clean_text_transformer(sample_text)}")
    print()
    
    # 2. Traditional Model Demo
    print("2. Traditional Model Demo")
    
    # Sample data
    texts = [
        "Scientists have published peer-reviewed research on climate change effects",
        "SHOCKING: Celebrity does something normal, doctors hate this trick!",
        "The Federal Reserve announced new interest rate policies today",
        "You won't believe this AMAZING secret that billionaires don't want you to know!",
        "University researchers found significant correlations in their data analysis"
    ]
    labels = [1, 0, 1, 0, 1]  # 1=Real, 0=Fake
    
    # Train TF-IDF classifier
    tfidf_model = TFIDFClassifier(max_features=100, min_df=1)
    tfidf_model.fit(texts, labels)
    
    # Make predictions
    test_texts = [
        "New research reveals important scientific findings",
        "CLICK HERE for this ONE WEIRD TRICK that will SHOCK you!"
    ]
    
    predictions = tfidf_model.predict(test_texts)
    probabilities = tfidf_model.predict_proba(test_texts)
    
    print("TF-IDF Predictions:")
    for text, pred, proba in zip(test_texts, predictions, probabilities):
        label = "Real" if pred == 1 else "Fake"
        confidence = max(proba)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print()

def demo_pipeline_usage():
    """Demonstrate full pipeline usage."""
    print("=== Demo: Full Pipeline Usage ===\n")
    
    print("Note: This demo uses minimal data for demonstration.")
    print("For full functionality, ensure you have the actual datasets.\n")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Demo with synthetic data (since real datasets might not be available)
    print("Creating synthetic dataset for demo...")
    
    import pandas as pd
    import numpy as np
    
    # Create synthetic data
    fake_texts = [
        "SHOCKING discovery that doctors don't want you to know about this miracle cure!",
        "BREAKING: Celebrity caught doing something completely outrageous and unbelievable!",
        "You won't believe what happened next in this AMAZING story that will blow your mind!",
        "URGENT: Government trying to hide this INCREDIBLE secret from the public!",
        "Scientists HATE this simple trick that will change your life forever!"
    ]
    
    real_texts = [
        "Researchers at Stanford University published findings in the Journal of Science today.",
        "The Federal Reserve announced a new monetary policy to address economic concerns.",
        "Climate scientists report new data on global temperature trends in peer-reviewed study.",
        "Government officials met to discuss infrastructure development and budget allocations.",
        "Medical researchers completed a clinical trial with promising results for treatment."
    ]
    
    # Combine and create dataframe
    all_texts = fake_texts + real_texts
    all_labels = [0] * len(fake_texts) + [1] * len(real_texts)  # 0=Fake, 1=Real
    
    # Create synthetic datasets
    synthetic_isot = pd.DataFrame({
        'text': all_texts * 10,  # Repeat for more data
        'label_binary': all_labels * 10
    })
    
    # Add preprocessed columns
    preprocessor = TextPreprocessor()
    synthetic_isot = preprocessor.preprocess_dataset(synthetic_isot, preprocessing_type='traditional')
    synthetic_isot = preprocessor.preprocess_dataset(synthetic_isot, preprocessing_type='transformer')
    
    # Store in detector
    detector.datasets['synthetic'] = synthetic_isot
    
    print(f"Created synthetic dataset with {len(synthetic_isot)} articles")
    
    # Train traditional models
    print("\nTraining traditional models...")
    try:
        traditional_results = detector.train_traditional_models('synthetic')
        print("Traditional models trained successfully!")
        
        for model_name, results in traditional_results.items():
            print(f"{model_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f} accuracy")
    
    except Exception as e:
        print(f"Traditional training failed: {e}")
    
    # Make predictions
    print("\nMaking predictions on new texts...")
    
    new_texts = [
        "Researchers conducted a controlled study with statistical significance testing.",
        "AMAZING breakthrough that will SHOCK you - click here for the SECRET!",
        "The university published peer-reviewed research in a scientific journal."
    ]
    
    for model_type in ['bow', 'tfidf']:
        if model_type in detector.traditional_models:
            try:
                results = detector.predict(new_texts, model_type=model_type)
                print(f"\n{model_type.upper()} Model Predictions:")
                
                for i, (text, label, conf) in enumerate(zip(new_texts, results['labels'], results['confidence'])):
                    print(f"{i+1}. {label} (confidence: {conf:.3f})")
                    print(f"   Text: {text[:60]}...")
            
            except Exception as e:
                print(f"Prediction failed for {model_type}: {e}")

def demo_advanced_features():
    """Demonstrate advanced features like calibration and evaluation."""
    print("=== Demo: Advanced Features ===\n")
    
    # Model Calibration Demo
    print("1. Model Calibration Demo")
    print("This would demonstrate confidence calibration for better uncertainty estimates.")
    print("See src/models/calibration.py for implementation details.\n")
    
    # Cross-domain Evaluation Demo  
    print("2. Cross-domain Evaluation Demo")
    print("This would show how models perform across different news domains.")
    print("See src/evaluation/metrics.py for implementation details.\n")
    
    # Ensemble Methods Demo
    print("3. Ensemble Methods Demo")
    print("This would demonstrate combining multiple models for better performance.")
    print("See src/models/ensemble.py for implementation details.\n")
    
    print("For full advanced features, run train_pipeline.py with real datasets.")

def main():
    """Run all demos."""
    print("Fake News Detection Pipeline Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        print("\n" + "="*50 + "\n")
        
        demo_pipeline_usage()
        print("\n" + "="*50 + "\n")
        
        demo_advanced_features()
        
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("1. Place real datasets (Fake.csv, True.csv, fake_or_real_news.csv) in data/raw/")
        print("2. Run: python train_pipeline.py")
        print("3. Explore the modular components in src/")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("This might be due to missing dependencies or datasets.")

if __name__ == "__main__":
    main()