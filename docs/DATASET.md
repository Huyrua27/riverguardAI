# Dataset & gán nhãn

## Nguồn dữ liệu
- **Public datasets:** floating debris / plastic waste trên mặt nước — dùng để pretrain & benchmark. Kê khai URL + license trong `configs/data.yaml`.
- **Dữ liệu tự thu thập:** video tại các đoạn kênh/sông thực tế, đa điều kiện — nắng gắt, nhiều mây, mưa, phản chiếu mạnh, các mức dòng chảy khác nhau.

## Gán nhãn
- Class (loại rác), bounding box và/hoặc segmentation mask.
- Gán trajectory/ID cho các đoạn video khi có thể.
- Chuẩn hoá **labeling guideline** để đảm bảo nhất quán giữa các thành viên.

## Thách thức dữ liệu đặc thù
Các false positive điển hình trên mặt nước cần xử lý chủ động:
`reflection` (phản chiếu), `glare` (loá sáng), `foam` (bọt), `wave` (sóng), `shadow` (bóng).

→ Áp dụng **augmentation** + **hard-negative mining** để phân biệt rác thật với nhiễu.

## Cấu trúc thư mục
```
data/
├── raw/         # video/ảnh gốc (gitignored)
├── processed/   # đã cắt frame, gán nhãn, chia split
└── external/    # public datasets tải về
```

Xem thêm tuân thủ dữ liệu tại [ETHICS.md](ETHICS.md).
