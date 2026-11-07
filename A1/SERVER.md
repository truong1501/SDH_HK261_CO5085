# A1 ưu tiên trên JupyterHub A100

Hạn A1.2: thứ Hai **09/11/2026, 23:59 giờ Việt Nam**, theo lịch trong repo. Không cần SSH hay UI demo. Push source mới lên Git từ máy cá nhân, clone trên JupyterHub rồi mở [notebooks/A1_JupyterHub.ipynb](notebooks/A1_JupyterHub.ipynb) tại vị trí gốc trong repo.

## Trình tự

1. Chọn kernel có CUDA/PyTorch/torchvision đúng với GPU server. Notebook không cài lại CUDA hệ thống.
2. Đặt `REPO`, `DATA_ROOT`, `OUT` đúng đường dẫn có quyền ghi và giữ được qua các phiên 12 giờ.
3. Chạy setup và tests, xem dry-run queue.
4. Chuyển checkpoint R3_s42 nếu muốn resume: `last.pt`, `best.pt`, `config.yaml`, `history.csv`. Chúng nằm trong A1/runs/R3_s42 ở Drive/máy cá nhân; file .pt không có trong Git. Nếu không chuyển, run chưa hoàn tất sẽ train lại từ đầu.
5. Chạy hàng đợi với `HOURS=11`, để khoảng một giờ cho tổng hợp và tải kết quả. Giới hạn mềm: dừng trước epoch mới nếu thời gian còn ít, không kill giữa lúc ghi checkpoint. Không bảo đảm đủ tất cả run trong 12 giờ; không tự giảm epoch.
6. Chạy tổng hợp và export. Tải archive kết quả nhỏ từ notebook và đưa lên Drive theo quy định công ty. Checkpoint resume có archive tùy chọn riêng; không commit Git.

Food-101 tải trực tiếp bằng torchvision/ETH hoặc dùng tar.gz đã upload qua cơ chế công ty cho phép. Không cần mount Google Drive, không cần token Drive trong notebook. Nếu dữ liệu server không được phép đưa ra ngoài, chỉ mang các metric/biểu đồ được phép; notebook không tự upload dữ liệu lên dịch vụ nào.

## Thứ tự hàng đợi

Trước hết seed 42: R2, R6, R4, R8, R3, R7, R1, R5. Sau đó seed 43/44 của các run pretrained. Run có metrics.json được bỏ qua; standard run còn dở resume last.pt. Tám run hiện đã xong là R2/R6 ở ba seed và R4/R8 ở seed 42; R3_s42 còn dở. Khi clone phải mang các metrics/config/history nhỏ đã có để runner biết run nào hoàn tất.

Không chạy sweep LR, X1–X8 hoặc A2 trên cùng GPU trước khi hoàn thành phần bắt buộc A1. Có đủ kết quả tám cấu hình seed 42 trước, rồi mới bổ sung độ biến thiên qua seed.

## CLI tương đương (không bắt buộc dùng terminal)

```bash
cd A1
python -m src.server --data_root /path/to/datasets --hours 11 --dry_run
python -u -m src.server --data_root /path/to/datasets --hours 11 --num_workers 4
python -m src.summarize --out_root runs --splits_dir splits --fig_dir figures
```

Exit code 75 nghĩa là đã hết ngân sách mềm, không phải đủ kết quả. Xem `runs/session_status.json` để biết hàng đợi còn lại. Run sau dùng lại OUT/SPLITS để tiếp tục. Nếu lỗi khác, sửa nguyên nhân trước khi chạy lại. File `metrics.json` chỉ được ghi sau khi test và benchmark xong, không dùng sự tồn tại của folder để coi run hoàn tất.
