# Model Card — RiverGuard AI

## Mục đích
Phát hiện, theo dõi động học và dự báo điểm nóng (hotspot) rác trôi nổi trên sông/kênh, phục vụ điều phối thu gom.

## Kiến trúc
Pipeline nhiều tầng: Detection (YOLO v8/v11 hoặc RT-DETR) → Tracking (ByteTrack/BoT-SORT) → Motion (optical flow: RAFT/Farnebäck) → Hotspot Prediction (grid-based temporal / ConvLSTM) → Decision (risk & priority scoring).

## Dữ liệu huấn luyện
Public floating-debris datasets + dữ liệu tự thu thập đa điều kiện. Xem [DATASET.md](DATASET.md).

## Chỉ số đánh giá
Xem [EVALUATION.md](EVALUATION.md). Báo cáo kèm baseline/ablation và uncertainty/confidence.

## Giới hạn đã biết
- Rác nhỏ / ở xa camera dễ bị bỏ sót → cần ảnh độ phân giải cao + tiling.
- Nhiễu mặt nước (reflection, glare, foam, wave, shadow) gây false positive.
- Domain shift khi đổi bối cảnh camera → cần fine-tuning / domain adaptation.
- Dự báo hotspot mang tính xác suất → luôn kèm confidence và ngưỡng cảnh báo.

## Sử dụng có trách nhiệm
Xem [ETHICS.md](ETHICS.md). Dữ liệu ẩn danh; công cụ AI được kê khai trong [PROMPT_LOG.md](PROMPT_LOG.md).
