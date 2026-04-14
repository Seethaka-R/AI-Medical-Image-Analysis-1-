# GitHub Publishing Strategy

## Initial Push
```bash
git init
git add .
git commit -m "feat: initial project scaffolding"
git branch -M main
git remote add origin https://github.com/<username>/AI-Medical-Image-Analysis.git
git push -u origin main
```

## Commit Plan (7 Days)
Day 1: feat: project scaffolding and requirements
Day 2: feat: CLAHE preprocessing pipeline and synthetic dataset
Day 3: feat: ResNet50 transfer learning model with Grad-CAM
Day 4: feat: two-phase trainer and evaluation suite
Day 5: feat: Flask REST API with Socket.IO real-time streaming
Day 6: feat: interactive dashboard with live charts and prediction
Day 7: docs: results screenshots and polished README

## Topics to Add
medicalimaging deeplearning transferlearning resnet50 gradcam flask
computer-vision tensorflow healthcare-ai xai python

## Do NOT Upload
- data/raw/ (link to Kaggle instead)
- models/saved/*.h5 (use GitHub Releases)
- venv/ __pycache__/
