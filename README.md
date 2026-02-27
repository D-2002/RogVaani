# RogVaani — Predictive. Preventive. Personalized.
AI for Bharat Hackathon | Team: Nested Loops

## Architecture
AWS ap-south-1 | SageMaker → Lambda → API Gateway → Next.js

## Model
Ridge regression (constrained temporal disaggregation)
Training: 2015–2023 Mumbai | Features: weather anomalies, AQI, search trends
See: notebooks/RogVaani_Model_Training_v4_final.ipynb

## Artifacts
Model artifacts stored in S3 (not committed — see feature_metadata.json for schema)
