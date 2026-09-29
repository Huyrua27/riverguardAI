# Evaluation & Ablation

## Bộ chỉ số

| Nhóm | Metrics |
|------|---------|
| Detection | Precision, Recall, mAP@50, mAP@50:95 |
| Tracking | IDF1, MOTA, HOTA |
| Prediction | MAE, RMSE, hotspot precision / recall |
| Early warning | Top-1 / Top-3 hotspot recall, lead time (thời gian cảnh báo sớm) |
| System | FPS, latency, GPU memory |

## Baseline & Ablation

Mục tiêu: **định lượng đóng góp của từng thành phần**, thay vì báo cáo một con số duy nhất — đúng tinh thần chấm điểm Bảng C.

1. Detection only
2. Detection + Tracking
3. Detection + Tracking + Motion
4. Detection + Tracking + Motion + Temporal Prediction
5. **Proposed:** + Hotspot Ranking / Decision Support

Chạy: `python scripts/evaluate.py --ablation <name>`.
