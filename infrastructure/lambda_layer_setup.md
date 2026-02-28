# Lambda Layer: rogvaani-sklearn

## Purpose
scikit-learn + dependencies for Lambda 2 (rogvaani-risk-inference)

## Build commands (run in AWS CloudShell)
pip install scikit-learn scipy joblib threadpoolctl numpy \
    --target sklearn_layer/python \
    --only-binary=:all: \
    --python-version 3.11 \
    --platform manylinux2014_x86_64 \
    --implementation cp -q

zip -r sklearn_layer.zip python/
aws s3 cp sklearn_layer.zip s3://rogvaani-data-lake/layers/sklearn_layer.zip

## Layer config
- Name: rogvaani-sklearn
- Runtime: Python 3.11
- S3 path: s3://rogvaani-data-lake/layers/sklearn_layer.zip

## Attached to
- rogvaani-risk-inference (Lambda 2)
```

Commit message:
```
feat: Lambda 2 working — risk inference, ward downscaling, heatmap output verified
fix: sklearn Lambda layer via CloudShell manylinux2014 build
