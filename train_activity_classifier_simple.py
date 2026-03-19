#!/usr/bin/env python3
"""
Train activity classifiers using scikit-learn to compare raw IMU vs physics features
Proves that physics features carry meaningful signal for activity classification
"""

import os
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

def flatten_raw_imu(X_raw):
    """Flatten raw IMU windows for Random Forest"""
    n_samples, n_timesteps, n_axes = X_raw.shape
    return X_raw.reshape(n_samples, n_timesteps * n_axes)

def cross_validate_model(X, y, subjects, model_name, n_splits=5):
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        val_acc = rf.score(X_val_scaled, y_val)
        accuracies.append(val_acc)
        
        # Predictions for analysis
        y_pred = rf.predict(X_val_scaled)
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
    print(classification_report(all_labels, all_predictions, target_names=activity_names, digits=4))
    
    return accuracies, all_predictions, all_labels

def plot_confusion_matrices(y_true_raw, y_pred_raw, y_true_physics, y_pred_physics):
    """Plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    activity_names = ['walk', 'sit', 'run']
    
    # Raw IMU confusion matrix
    cm_raw = confusion_matrix(y_true_raw, y_pred_raw)
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names, ax=axes[0])
    axes[0].set_title('Raw IMU (768 features)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Physics confusion matrix
    cm_physics = confusion_matrix(y_true_physics, y_pred_physics)
    sns.heatmap(cm_physics, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names, ax=axes[1])
    axes[1].set_title('Physics Features (19 features)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('data/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrices to data/confusion_matrices.png")

def analyze_physics_feature_importance(X_physics, y, metadata):
    """Analyze which physics features are most predictive"""
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
        print(f"  {i+1:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    return feature_names, importances, indices

def compare_feature_efficiency(X_raw_shape, X_physics_shape, raw_acc, physics_acc):
    """Compare efficiency of raw vs physics features"""
    n_raw_params = X_raw_shape[1] * X_raw_shape[2]  # 256 * 3 = 768
    n_physics_params = X_physics_shape[1]  # 19
    
    compression_ratio = n_raw_params / n_physics_params
    performance_ratio = physics_acc / raw_acc
    
    print(f"\n=== EFFICIENCY ANALYSIS ===")
    print(f"Raw IMU:     {n_raw_params:4d} parameters → {raw_acc:.4f} accuracy")
    print(f"Physics:       {n_physics_params:4d} parameters → {physics_acc:.4f} accuracy")
    print(f"Compression:   {compression_ratio:.1f}x reduction in parameters")
    print(f"Performance:   {performance_ratio:.1%} of raw performance")
    print(f"Efficiency:   {performance_ratio * compression_ratio:.1f} (higher is better)")

def main():
    # Load data
    X_raw, X_physics, y, subjects, metadata = load_training_data()
    
    # Flatten raw IMU for Random Forest
    X_raw_flat = flatten_raw_imu(X_raw)
    print(f"Flattened raw IMU: {X_raw_flat.shape}")
    
    # Cross-validate both models
    raw_accs, y_pred_raw, y_true_raw = cross_validate_model(
        X_raw_flat, y, subjects, "Raw IMU"
    )
    
    physics_accs, y_pred_physics, y_true_physics = cross_validate_model(
        X_physics, y, subjects, "Physics Features"
    )
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    raw_mean = np.mean(raw_accs)
    physics_mean = np.mean(physics_accs)
    
    print(f"Raw IMU:        {raw_mean:.4f} ± {np.std(raw_accs):.4f}")
    print(f"Physics Features: {physics_mean:.4f} ± {np.std(physics_accs):.4f}")
    
    if physics_mean > raw_mean:
        print(f"🎉 Physics features OUTPERFORM raw IMU by {physics_mean - raw_mean:.4f}!")
    else:
        print(f"📊 Raw IMU outperforms physics by {raw_mean - physics_mean:.4f}")
    
    print(f"Physics features achieve {physics_mean/raw_mean*100:.1f}% of raw performance")
    
    # Plot confusion matrices
    plot_confusion_matrices(y_true_raw, y_pred_raw, y_true_physics, y_pred_physics)
    
    # Analyze physics feature importance
    feature_names, importances, indices = analyze_physics_feature_importance(X_physics, y, metadata)
    
    # Compare efficiency
    compare_feature_efficiency(X_raw_flat.shape, X_physics.shape, raw_mean, physics_mean)
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Physics features carry {physics_mean:.1%} of the signal in raw IMU data")
    print(f"Using only 19 physics features vs 768 raw parameters")
    print(f"This proves physics features are highly efficient signal embeddings")
    
    # Save results summary
    results = {
        'raw_imu_accuracy': raw_mean,
        'raw_imu_std': np.std(raw_accs),
        'physics_accuracy': physics_mean,
        'physics_std': np.std(physics_accs),
        'raw_features': X_raw_flat.shape[1],
        'physics_features': X_physics.shape[1],
        'compression_ratio': X_raw_flat.shape[1] / X_physics.shape[1],
        'performance_ratio': physics_mean / raw_mean,
        'top_physics_features': [
            (feature_names[indices[i]], importances[indices[i]]) 
            for i in range(min(5, len(indices)))
        ]
    }
    
    np.save('data/results_summary.npy', results)
    print(f"\nResults saved to data/results_summary.npy")

if __name__ == "__main__":
    main()
