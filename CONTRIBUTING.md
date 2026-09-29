# Đóng góp

## Quy ước commit
Dùng [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `exp:` (experiment).

## Nhánh
- `main` — luôn chạy được, được bảo vệ.
- `dev` — tích hợp.
- `feat/<tên>`, `exp/<tên>` — nhánh làm việc, PR về `dev`.

## Trước khi mở PR
```bash
pre-commit run --all-files   # ruff + black + mypy
pytest                        # tests phải xanh
```
Mỗi PR cần ≥1 review chéo (yêu cầu minh bạch commit history của Thể lệ).

## Reproducibility
- Mọi experiment ghi lại config + seed + commit hash trong `outputs/<run_id>/`.
- Không commit weight/data nặng vào git; dùng DVC hoặc release assets.
- Cập nhật `docs/PROMPT_LOG.md` khi dùng công cụ AI hỗ trợ code.
