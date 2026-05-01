# Segmentation Tasks

This package contains segmentation task plugins and shared segmentation helpers.

`segmentation_adapter.py` defines `SegmentationAdapter`, an intermediate base class for segmentation-style tiling adapters:

```text
BaseAdapter -> SegmentationAdapter -> ConcreteTaskAdapter
```

It provides reusable foreground-mask, GT-presence, and prediction-vs-GT difficulty helpers while leaving task-specific CSV columns to concrete adapters.
