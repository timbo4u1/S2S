#!/usr/bin/env python3
"""
Train activity classifiers to compare raw IMU vs physics features
Proves that physics features carry meaningful signal for activity classification
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_data(data_dir="data"):
    """Load the extracted training data"""
    print("Loading training data...")
    
    X_raw = np.load(os.path.join(data_dir, 'training_X_raw.npy'))
    X_physics = np.load(os.path.join(data_dir, 'training_X_physics.npy'))
    y = np.load(os.path.join(data_dir, 'training_y.npy'))
    subjects = np.load(os.path.join(data_dir, 'training_subjects.npy'))
    metadata = np.load(os.path.join(data_dir, 'training_metadata.npy'), allow_pickle=True).item()
    
    print(f"Raw IMU: {X_raw.shape}")
    print(f"Physics features: {X_physics.shape}")
    print(f"Labels: {y.shape}")
    print(f"Subjects: {subjects.shape}")
    
    return X_raw, X_physics, y, subjects, metadata

def build_cnn_model(input_shape, n_classes=3):
    """Build a simple CNN for raw IMU data"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 7, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_physics_model(input_shape, n_classes=3):
    """Build a simple classifier for physics features"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def cross_validate_model(X, y, subjects, model_builder, model_name, n_splits=5):
    """Perform subject-wise cross-validation"""
    print(f"\n=== {model_name} Cross-Validation ===")
    
    group_kfold = GroupKFold(n_splits=n_splits)
    accuracies = []
    all_predictions = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, subjects)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build and train model
        if model_name == "CNN Raw IMU":
            model = model_builder(input_shape=(X_train.shape[1], X_train.shape[2]))
        else:
            model = model_builder(input_shape=(X_train.shape[1],))
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64,
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(val_acc)
        
        # Predictions for analysis
        y_pred = np.argmax(model.predict(X_val), axis=1)
        all_predictions.extend(y_pred)
        all_labels.extend(y_val)
        
        print(f"  Val accuracy: {val_acc:.4f}")
    
    # Overall results
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{model_name} Results:")
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best fold: {max(accuracies):.4f}")
    print(f"  Worst fold: {min(accuracies):.4f}")
    
    # Classification report
    activity_names = ['walk', 'sit', 'run']
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=activity_names))
    
    return accuracies, all_predictions, all_labels

def plot_confusion_matrices(y_true_cnn, y_pred_cnn, y_true_physics, y_pred_physics):
    """Plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    activity_names = ['walk', 'sit', 'run']
    
    # CNN confusion matrix
    cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names, ax=axes[0])
    axes[0].set_title('CNN Raw IMU')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Physics confusion matrix
    cm_physics = confusion_matrix(y_true_physics, y_pred_physics)
    sns.heatmap(cm_physics, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names, ax=axes[1])
    axes[1].set_title('Physics Features')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('data/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrices to data/confusion_matrices.png")

def analyze_physics_feature_importance(X_physics, y, metadata):
    """Analyze which physics features are most predictive"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    print(f"\n=== Physics Feature Importance Analysis ===")
    
    # Train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_physics, y)
    
    # Feature importance
    feature_names = metadata['physics_feature_names']
    importances = rf.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 most important physics features:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return feature_names, importances, indices

def main():
    # Load data
    X_raw, X_physics, y, subjects, metadata = load_training_data()
    
    # Normalize raw IMU data
    print("Normalizing raw IMU data...")
    X_raw_norm = (X_raw - np.mean(X_raw, axis=(0, 1), keepdims=True)) / np.std(X_raw, axis=(0, 1), keepdims=True)
    
    # Normalize physics features
    X_physics_norm = (X_physics - np.mean(X_physics, axis=0)) / np.std(X_physics, axis=0)
    
    # Cross-validate both models
    cnn_accs, y_pred_cnn, y_true_cnn = cross_validate_model(
        X_raw_norm, y, subjects, build_cnn_model, "CNN Raw IMU"
    )
    
    physics_accs, y_pred_physics, y_true_physics = cross_validate_model(
        X_physics_norm, y, subjects, build_physics_model, "Physics Features"
    )
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    print(f"CNN Raw IMU:    {np.mean(cnn_accs):.4f} ± {np.std(cnn_accs):.4f}")
    print(f"Physics Features: {np.mean(physics_accs):.4f} ± {np.std(physics_accs):.4f}")
    
    cnn_mean = np.mean(cnn_accs)
    physics_mean = np.mean(physics_accs)
    
    if physics_mean > cnn_mean:
        print(f"🎉 Physics features OUTPERFORM raw CNN by {physics_mean - cnn_mean:.4f}!")
    else:
        print(f"📊 Raw CNN outperforms physics by {cnn_mean - physics_mean:.4f}")
    
    print(f"Physics features achieve {physics_mean/cnn_mean*100:.1f}% of CNN performance")
    
    # Plot confusion matrices
    plot_confusion_matrices(y_true_cnn, y_pred_cnn, y_true_physics, y_pred_physics)
    
    # Analyze physics feature importance
    feature_names, importances, indices = analyze_physics_feature_importance(X_physics_norm, y, metadata)
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Physics features carry {physics_mean:.1%} of the signal in raw IMU data")
    print(f"Using only 19 physics features vs 768 raw parameters (256×3)")
    print(f"Compression ratio: {768/19:.1f}x reduction with {physics_mean:.1%} performance preserved")

if __name__ == "__main__":
    main()
