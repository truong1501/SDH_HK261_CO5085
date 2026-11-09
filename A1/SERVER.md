# A1 ưu tiên trên JupyterHub A100

Hạn A1.2: thứ Hai **09/11/2026, 23:59 giờ Việt Nam**, theo lịch trong repo. Không cần SSH hay UI demo. Push source mới lên Git từ máy cá nhân, clone trên JupyterHub rồi mở [notebooks/A1_JupyterHub.ipynb](notebooks/A1_JupyterHub.ipynb) tại vị trí gốc trong repo.

## Trình tự

Notebook chạy **trong kernel** (gọi thẳng `src.train.run_one`), không dùng `subprocess`/CLI con.

1. Chọn kernel có GPU. Notebook chỉ `%pip install` gói còn thiếu (timm, pyyaml...), không cài lại torch/CUDA.
2. Cell 1: sửa `DATA_ROOT`, `OUT` nếu cần (A1 tự nhận khi mở notebook trong `A1/notebooks`).
3. Cell 2 tự chọn `num_workers` theo CPU và `/dev/shm`; nếu shm nhỏ (container 64 MB) sẽ chuyển sharing strategy sang `file_system` để tránh "Bus error"/DataLoader worker chết.
4. Muốn resume R3_s42 thì chép `last.pt`, `best.pt`, `config.yaml`, `history.csv` vào `OUT/R3_s42/` (file .pt không có trong Git).
5. Cell 5 train. Dừng bằng Interrupt kernel; checkpoint ghi atomic (file tạm + rename) nên epoch đã lưu không bị hỏng. Chạy lại cell 5 là resume. `HOURS` là giới hạn mềm tại ranh giới epoch.
6. Cell 6–7 tổng hợp và xuất archive nhỏ; cell 8 (tuỳ chọn) đóng gói checkpoint.

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
