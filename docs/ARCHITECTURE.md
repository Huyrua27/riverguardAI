# Kiến trúc hệ thống

## Luồng xử lý chính

```
Camera/Drone
  → Video Processing
  → Waste Detection
  → Classification / Segmentation
  → Multi-Object Tracking
  → Motion Analysis
  → (Historical / Environmental Data)
  → Hotspot Prediction
  → Pollution Risk Map
  → Decision Dashboard
```

## Ba tầng

| Tầng | Thành phần | Vai trò |
|------|------------|---------|
| **Perception** | Detection, Segmentation, Tracking | Biến pixel thành đối tượng có ID và quỹ đạo |
| **Reasoning**  | Motion Analysis, Hotspot Prediction | Hiểu động học, dự báo nơi rác sắp tập trung |
| **Decision**   | Risk Scoring, Dashboard, Alert | Xếp hạng ưu tiên, cảnh báo, trực quan hoá |

## Triển khai (edge/cloud)

- **Perception** có thể chạy tại **edge** (gần camera) để giảm băng thông.
- **Reasoning** + **Decision** chạy trên **server/cloud**, cung cấp dashboard qua web.

## Ánh xạ code

| Tầng | Package |
|------|---------|
| Perception | `riverguard.detection`, `riverguard.tracking` |
| Reasoning | `riverguard.motion`, `riverguard.prediction` |
| Decision | `riverguard.decision`, `riverguard.dashboard` |

Orchestration: `riverguard.pipeline`.
