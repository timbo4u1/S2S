#!/usr/bin/env python3
"""
Hybrid Classifier: Physics Features + Raw IMU
Combines 19 physics features with 768 raw IMU parameters
Target: Beat both physics-only (70.5%) and raw-only (79.6%)
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

def create_hybrid_features(X_raw_flat, X_physics):
    """Combine raw IMU and physics features"""
    # Concatenate raw IMU (768) + physics features (19) = 787 total features
    X_hybrid = np.concatenate([X_raw_flat, X_physics], axis=1)
    print(f"Hybrid features shape: {X_hybrid.shape}")
    return X_hybrid

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

def plot_comparison_confusion_matrices(y_true_raw, y_pred_raw, y_true_physics, y_pred_physics, y_true_hybrid, y_pred_hybrid):
    """Plot confusion matrices for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    # Hybrid confusion matrix
    cm_hybrid = confusion_matrix(y_true_hybrid, y_pred_hybrid)
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names, ax=axes[2])
    axes[2].set_title('Hybrid (787 features)')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('data/hybrid_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrices to data/hybrid_confusion_matrices.png")

def analyze_feature_importance_hybrid(rf, metadata):
    """Analyze which features are most important in hybrid model"""
    print(f"\n=== Hybrid Feature Importance Analysis ===")
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Split into raw and physics parts
    n_raw = 768
    n_physics = 19
    
    raw_importances = importances[:n_raw]
    physics_importances = importances[n_raw:]
    
    # Top raw features
    raw_top_indices = np.argsort(raw_importances)[-10:][::-1]
    print("Top 10 Raw IMU Features:")
    for i, idx in enumerate(raw_top_indices):
        axis = idx % 3
        timestep = idx // 3
        print(f"  {i+1:2d}. Axis {axis} at timestep {timestep}: {raw_importances[idx]:.4f}")
    
    # Physics features
    physics_names = metadata['physics_feature_names']
    physics_top_indices = np.argsort(physics_importances)[::-1]
    print("\nPhysics Features (all 19):")
    for i, idx in enumerate(physics_top_indices):
        print(f"  {i+1:2d}. {physics_names[idx]:30s}: {physics_importances[idx]:.4f}")
    
    # Summary
    total_raw_importance = np.sum(raw_importances)
    total_physics_importance = np.sum(physics_importances)
    print(f"\nImportance Distribution:")
    print(f"  Raw IMU (768 features): {total_raw_importance:.3f} ({total_raw_importance*100:.1f}%)")
    print(f"  Physics (19 features): {total_physics_importance:.3f} ({total_physics_importance*100:.1f}%)")
    print(f"  Physics efficiency: {total_physics_importance/19:.4f} per feature")
    print(f"  Raw efficiency: {total_raw_importance/768:.4f} per feature")
    
    return raw_importances, physics_importances

def main():
    # Load data
    X_raw, X_physics, y, subjects, metadata = load_training_data()
    
    # Flatten raw IMU for Random Forest
    X_raw_flat = flatten_raw_imu(X_raw)
    print(f"Flattened raw IMU: {X_raw_flat.shape}")
    
    # Create hybrid features
    X_hybrid = create_hybrid_features(X_raw_flat, X_physics)
    
    # Cross-validate all three models
    raw_accs, y_pred_raw, y_true_raw = cross_validate_model(X_raw_flat, y, subjects, "Raw IMU")
    physics_accs, y_pred_physics, y_true_physics = cross_validate_model(X_physics, y, subjects, "Physics Features")
    hybrid_accs, y_pred_hybrid, y_true_hybrid = cross_validate_model(X_hybrid, y, subjects, "Hybrid Model")
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    raw_mean = np.mean(raw_accs)
    physics_mean = np.mean(physics_accs)
    hybrid_mean = np.mean(hybrid_accs)
    
    print(f"Raw IMU:        {raw_mean:.4f} ± {np.std(raw_accs):.4f}")
    print(f"Physics:        {physics_mean:.4f} ± {np.std(physics_accs):.4f}")
    print(f"Hybrid:         {hybrid_mean:.4f} ± {np.std(hybrid_accs):.4f}")
    
    # Calculate improvements
    hybrid_vs_raw = hybrid_mean - raw_mean
    hybrid_vs_physics = hybrid_mean - physics_mean
    
    print(f"\nHybrid Improvements:")
    print(f"  vs Raw IMU:     {hybrid_vs_raw:+.4f} ({hybrid_vs_raw/raw_mean*100:+.1f}%)")
    print(f"  vs Physics:     {hybrid_vs_physics:+.4f} ({hybrid_vs_physics/physics_mean*100:+.1f}%)")
    
    # Check if target achieved
    if hybrid_mean > raw_mean and hybrid_mean > physics_mean:
        print(f"\n🎉 SUCCESS: Hybrid model beats both baselines!")
        print(f"   Beats raw IMU by {hybrid_vs_raw*100:.2f}%")
        print(f"   Beats physics by {hybrid_vs_physics*100:.2f}%")
    elif hybrid_mean > raw_mean:
        print(f"\n✅ PARTIAL: Hybrid beats raw IMU but not physics")
    elif hybrid_mean > physics_mean:
        print(f"\n✅ PARTIAL: Hybrid beats physics but not raw IMU")
    else:
        print(f"\n❌ TARGET NOT MET: Hybrid doesn't beat either baseline")
    
    # Plot confusion matrices
    plot_comparison_confusion_matrices(y_true_raw, y_pred_raw, y_true_physics, y_pred_physics, y_true_hybrid, y_pred_hybrid)
    
    # Train final hybrid model for feature importance
    print(f"\n=== Training Final Hybrid Model for Feature Analysis ===")
    scaler = StandardScaler()
    X_hybrid_scaled = scaler.fit_transform(X_hybrid)
    
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final.fit(X_hybrid_scaled, y)
    
    # Analyze feature importance
    raw_importances, physics_importances = analyze_feature_importance_hybrid(rf_final, metadata)
    
    # Save results
    results = {
        'raw_accuracy': raw_mean,
        'raw_std': np.std(raw_accs),
        'physics_accuracy': physics_mean,
        'physics_std': np.std(physics_accs),
        'hybrid_accuracy': hybrid_mean,
        'hybrid_std': np.std(hybrid_accs),
        'hybrid_vs_raw_improvement': hybrid_vs_raw,
        'hybrid_vs_physics_improvement': hybrid_vs_physics,
        'target_achieved': hybrid_mean > raw_mean and hybrid_mean > physics_mean,
        'raw_features': X_raw_flat.shape[1],
        'physics_features': X_physics.shape[1],
        'hybrid_features': X_hybrid.shape[1],
        'raw_importance_total': float(np.sum(raw_importances)),
        'physics_importance_total': float(np.sum(physics_importances))
    }
    
    np.save('data/hybrid_results.npy', results)
    print(f"\nResults saved to data/hybrid_results.npy")
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"Hybrid model achieves {hybrid_mean:.1%} accuracy")
    print(f"Using {X_hybrid.shape[1]} features (768 raw + 19 physics)")
    print(f"Physics features contribute {np.sum(physics_importances)*100:.1f}% of predictive power")
    print(f"Target achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")

if __name__ == "__main__":
    main()
