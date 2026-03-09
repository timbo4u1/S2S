# Hybrid CNN Architecture Plan

## 📊 Real Dataset Analysis Results

**PTT-PPG Dataset (22 subjects, 500Hz):**
- Total windows: 63,314
- **GOLD**: 0 (0.0%)
- **SILVER**: 482 (0.8%)
- **BRONZE**: 9,843 (15.5%)
- **REJECTED**: 52,989 (83.7%)

**NinaPro DB5**: Loading failed (need to fix zip extraction)

## 🎯 Key Insights

1. **Physics Engine is Very Conservative**: 83.7% REJECTED rate
2. **Ambiguous Cases**: 16.3% (SILVER + BRONZE) = 10,325 windows
3. **Perfect for Hybrid**: Physics handles certainty, CNN resolves ambiguity

## 🏗️ Hybrid Architecture Design

### Phase 1: Physics-First Router
```python
def hybrid_certify(window):
    physics_result = physics_engine.certify(window)
    
    # Physics has certainty on extremes
    if physics_result['tier'] == 'REJECTED':
        return {'tier': 'REJECTED', 'method': 'physics', 'confidence': 0.95}
    elif physics_result['overall_score'] >= 90:  # Near-GOLD
        return {'tier': 'GOLD', 'method': 'physics', 'confidence': 0.90}
    
    # Ambiguous case - use CNN
    return cnn_classifier.predict(window)
```

### Phase 2: CNN Ambiguity Resolver

**Training Data:**
- Only SILVER (482) + BRONZE (9,843) windows = 10,325 samples
- **Problem**: SILVER severely underrepresented (4.7% vs 95.3% BRONZE)

**Architecture:**
```python
class AmbiguityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.physics_score = nn.Linear(1, 32)  # Physics score as feature
        self.classifier = nn.Linear(256 + 32, 2)  # SILVER vs BRONZE
        
    def forward(self, x, physics_score):
        # Standard CNN path
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x).squeeze(-1)
        
        # Add physics score as additional feature
        p = F.relu(self.physics_score(physics_score.unsqueeze(-1)))
        x = torch.cat([x, p], dim=1)
        
        return self.classifier(x)
```

### Phase 3: Training Strategy

**Data Augmentation for SILVER:**
- Oversample SILVER windows 20x to balance classes
- Add small noise to create variations
- Use focal loss to handle class imbalance

**Training Regimen:**
```python
# Balanced batches
silver_loader = DataLoader(silver_windows, batch_size=32, shuffle=True)
bronze_loader = DataLoader(bronze_windows, batch_size=32, shuffle=True)

for epoch in range(100):
    for silver_batch, bronze_batch in zip(silver_loader, bronze_batch):
        # Oversample silver: duplicate 20 times
        silver_batch = silver_batch.repeat(20, 1, 1)
        
        # Combined batch
        x = torch.cat([silver_batch, bronze_batch])
        y = torch.cat([torch.zeros(len(silver_batch)), torch.ones(len(bronze_batch))])
        
        # Forward pass with physics scores
        physics_scores = compute_physics_score(x)
        logits = model(x, physics_scores)
        
        # Focal loss for class imbalance
        loss = focal_loss(logits, y)
```

### Phase 4: Hybrid Decision Logic

```python
def hybrid_certify_final(window):
    # Step 1: Physics evaluation
    physics_result = physics_engine.certify(window)
    score = physics_result['overall_score']
    
    # Step 2: Physics certainty rules
    if score < 30:  # Clearly bad
        return {'tier': 'REJECTED', 'method': 'physics', 'confidence': 0.95}
    elif score > 85:  # Clearly good
        return {'tier': 'GOLD', 'method': 'physics', 'confidence': 0.90}
    
    # Step 3: CNN for ambiguous cases
    cnn_input = torch.tensor(window).unsqueeze(0)
    physics_score_tensor = torch.tensor([[score]])
    
    with torch.no_grad():
        logits = cnn_model(cnn_input, physics_score_tensor)
        probs = F.softmax(logits, dim=1)
        
        if probs[0][0] > 0.6:  # SILVER confidence
            return {'tier': 'SILVER', 'method': 'hybrid', 'confidence': probs[0][0].item()}
        else:  # BRONZE
            return {'tier': 'BRONZE', 'method': 'hybrid', 'confidence': probs[0][1].item()}
```

## 🎯 Expected Performance

**Benefits:**
1. **Physics Certainty**: 83.7% of cases handled by physics alone
2. **CNN Focus**: Only trains on 16.3% ambiguous cases
3. **Higher Accuracy**: CNN doesn't need to learn extreme cases
4. **Explainability**: Clear decision path for each tier

**Target Metrics:**
- Overall accuracy: >90% (vs 52% pure CNN)
- SILVER precision: >80% (vs current 0.8%)
- REJECTED recall: >95% (physics handles this)

## 📋 Implementation Steps

1. **Fix NinaPro Loading** - Extract zip files properly
2. **Extract Training Data** - Get SILVER/BRONZE windows with physics scores
3. **Balance Dataset** - Oversample SILVER class
4. **Train CNN** - Use focal loss and physics score features
5. **Build Hybrid Router** - Implement decision logic
6. **Benchmark** - Compare vs pure physics and pure CNN

## 🔧 Next Actions

1. Fix NinaPro zip extraction in `count_real_datasets.py`
2. Create `extract_training_data.py` to prepare SILVER/BRONZE dataset
3. Implement `train_ambiguity_cnn.py`
4. Build `hybrid_certifier.py` with routing logic
5. Benchmark all three approaches

This hybrid approach leverages the physics engine's strength in certainty while using CNN only where it's needed most.
