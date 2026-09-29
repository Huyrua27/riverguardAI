# RiverGuard AI

> Hệ thống thị giác máy tính giám sát, theo dõi động học và **dự báo điểm nóng (hotspot)** rác thải trôi nổi trên sông.
> _Cuộc thi Sáng tạo trẻ Quốc gia trong lĩnh vực Trí tuệ nhân tạo 2026 — Bảng C._

[![CI](https://github.com/Huyrua27/riverguardAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Huyrua27/riverguardAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

RiverGuard AI vận hành theo triết lý **Observe → Understand → Predict → Act**: không chỉ trả lời *"ở đâu đang có rác"* mà *"ở đâu sắp có rác và nên xử lý chỗ nào trước"*.

---

## Pipeline

```
Camera/Drone → Video Processing → Waste Detection → Classification/Segmentation
   → Multi-Object Tracking → Motion Analysis → (Historical/Environmental Data)
   → Hotspot Prediction → Pollution Risk Map → Decision Dashboard
```

| Tầng | Module | Vai trò |
|------|--------|---------|
| **Perception** | `detection`, `tracking` | Biến pixel thành đối tượng có ID và quỹ đạo |
| **Reasoning**  | `motion`, `prediction` | Hiểu động học, dự báo nơi rác sắp tập trung |
| **Decision**   | `decision`, `dashboard` | Xếp hạng ưu tiên, cảnh báo, trực quan hoá |

## Quickstart

```bash
# 1. Môi trường
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dashboard,dev]"

# 2. Chạy pipeline end-to-end (Observe→Understand→Predict→Act)
#    Source có thể là ẢNH, THƯ MỤC ảnh, VIDEO, camera index hoặc stream URL.
#    Mặc định dùng YOLO-World (open-vocabulary) — chạy ZERO-SHOT, không cần train;
#    weights tự tải lần đầu. Sinh overlay + Risk Map + JSON + report HTML.
python -m riverguard.pipeline --source data/raw
#   → outputs/vis/*.jpg  (bbox + ID + class + hotspot markers + heatmap)
#   → outputs/results.json, outputs/report.html

# 3. Xử lý video (tracking + motion + hotspot theo thời gian)
python -m riverguard.pipeline --source data/raw/input1.mp4 --stride 5 \
    --output-dir outputs/video/input1
#   → outputs/video/input1/: input1_annotated.mp4 (H.264) + base_frame.jpg
#                            + risk_map.jpg + results.json

# 4. Decision Support Dashboard (sản phẩm demo)
#    Khai báo camera (tên, trạng thái, toạ độ GIS, hướng) trong configs/cameras.yaml
python -m riverguard.dashboard.app --output-dir outputs   # → http://127.0.0.1:8000
#    - Vận hành: Live camera (overlay box/ID/quỹ đạo) · Hotspot Alert + "Vì sao?"
#                · Pollution Risk Map (góc camera / GIS) · Hàng đợi ưu tiên toàn mạng
#    - Phân tích: so sánh camera, lịch sử hotspot, xu hướng, xuất CSV/JSON
#    Không cần server: python -m riverguard.dashboard.build → mở outputs/dashboard.html
#    (khi mở trực tiếp, bản đồ GIS cần Internet; các phần khác chạy offline)

# 5. Ablation matrix (đóng góp từng tầng — xem docs/EVALUATION.md)
python scripts/evaluate.py --source data/raw
#    Ablation offline trên detection đã lưu (tracker A/B/C, predictor cũ/mới; không cần GPU):
python scripts/ablation_offline.py --outputs outputs   # → outputs/ablation_offline.json

# (tuỳ chọn) đổi sang detector fine-tuned: sửa configs/default.yaml →
#   detection.model: yolo  + weights: <đường dẫn .pt đã train>
```

## Cấu trúc repo

```
riverguard-ai/
├── configs/            # YAML config: model, data, pipeline, tracker
├── data/               # raw / processed / external (gitignored)
├── docs/               # kiến trúc, dataset, evaluation, model card, ethics
├── notebooks/          # EDA, phân tích ablation
├── scripts/            # download data, train, eval, export
├── src/riverguard/     # source code (installable package)
│   ├── detection/      # YOLO / RT-DETR
│   ├── tracking/       # ByteTrack / BoT-SORT
│   ├── motion/         # optical flow, trajectory
│   ├── prediction/     # hotspot forecasting (ConvLSTM / grid-based)
│   ├── decision/       # risk & priority scoring
│   ├── dashboard/      # web dashboard + alerts
│   ├── data/           # dataset, loader, augmentation
│   └── utils/          # metrics, io, viz, logging
├── tests/              # unit + integration tests
├── models/             # weights (gitignored, dùng DVC/release)
└── outputs/            # kết quả chạy (gitignored)
```

## Tài liệu

- [Kiến trúc hệ thống](docs/ARCHITECTURE.md)
- [Dataset & gán nhãn](docs/DATASET.md)
- [Evaluation & ablation](docs/EVALUATION.md)
- [Roadmap](docs/ROADMAP.md)
- [Model Card](docs/MODEL_CARD.md)
- [Đạo đức & tuân thủ dữ liệu](docs/ETHICS.md)
- [Prompt Log (kê khai công cụ AI)](docs/PROMPT_LOG.md)
- [Đóng góp](CONTRIBUTING.md)

## Team (Bảng C — tối đa 03 thành viên)

| Vai trò | Phụ trách |
|---------|-----------|
| TV1 — Perception Lead (Đội trưởng) | Detection, Segmentation, Tracking |
| TV2 — Reasoning & Data Lead | Dataset, Motion, Hotspot Prediction, Evaluation |
| TV3 — Product & System Lead | Dashboard, Deploy, Báo cáo, Prompt Log |

## License

MIT — xem [LICENSE](LICENSE). Dữ liệu và pretrained model của bên thứ ba tuân theo giấy phép gốc tương ứng.
