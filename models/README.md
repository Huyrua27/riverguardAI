# models/

Weights **không** commit vào git (xem `.gitignore`). Quản lý qua GitHub Release assets hoặc tải tự động.

Detector mặc định là **YOLO-World-S** (zero-shot). Nếu `models/detection/yolov8s-world.pt` chưa có,
Ultralytics tự tải về ở lần chạy đầu (kèm trọng số CLIP vào `weights/`, cũng đã được gitignore).

```
models/
└── detection/
    ├── yolov8s-world.pt     # tự tải — YOLO-World-S (Ultralytics, AGPL-3.0)
    └── <fine-tuned>.pt      # kế hoạch: YOLO-seg fine-tune trên RiSID
```
