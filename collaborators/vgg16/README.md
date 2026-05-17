# VGG16 Collaborator Module

Fine-tuned VGG16 (torchvision) on HAM10000, 7-class skin lesion classification.

**Author:** Team member contribution  
**Architecture:** VGG16 (pretrained ImageNet → 2-stage fine-tune on HAM10000)  
**Input:** 256×256 resize → 224×224 center crop, ImageNet normalization  
**Labels (order):** `['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']`

## Model File

The trained checkpoint (`vgg16.pt`) is distributed via the GitHub Release page — it is too large for git history.

Download from: **Releases → vgg16-checkpoint** and place at:
```
collaborators/vgg16/vgg16.pt
```

The checkpoint should be saved with:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 7,
    'dx_labels': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
}, 'vgg16.pt')
```

> Note: labels remapped to the project's canonical alphabetical order on load.

## Files

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters and paths |
| `dataset.py` | HAM10000 dataset class |
| `imbalance.py` | Class-weight / sampler utilities |
| `train.py` | 2-stage fine-tune script |
| `evaluate.py` | 7-class evaluation metrics |
| `binary_eval.py` | Benign/malignant binary evaluation |
