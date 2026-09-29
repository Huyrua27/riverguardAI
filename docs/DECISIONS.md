# Architecture Decision Records (ADR)

Ghi lại các quyết định kỹ thuật quan trọng để bảo vệ đề tài trước hội đồng.

## ADR-001 — Ghép trọn pipeline thay vì tối ưu detection đơn lẻ
**Bối cảnh:** Nhiều giải pháp hiện có chỉ trả lời "có rác hay không".
**Quyết định:** Xây pipeline Observe→Understand→Predict→Act, đặt novelty ở tổ hợp *tracking động học + dự báo hotspot + hỗ trợ quyết định*.
**Hệ quả:** Cần đánh giá bằng ablation để định lượng đóng góp từng thành phần.

## ADR-002 — Baseline hotspot prediction dạng grid-based
**Bối cảnh:** ConvLSTM cần nhiều dữ liệu chuỗi thời gian có nhãn.
**Quyết định:** MVP dùng grid-based temporal accumulator (dễ giải thích, ít dữ liệu), giữ ConvLSTM là hướng nâng cấp.
**Hệ quả:** Ưu tiên chạy được đầu-cuối cho Vòng Khu vực.

## ADR-003 — Edge/cloud split
**Quyết định:** Perception ở edge, Reasoning+Decision ở cloud.
**Hệ quả:** Giảm băng thông, dashboard truy cập qua web.
