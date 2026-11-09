# A2 — Kế hoạch cải tiến sau baseline ip2p_v1

## Ưu tiên và kết quả đã có

Ưu tiên GPU cho A1 trước hạn 09/11/2026. A2.2 theo lịch repo là 16/11/2026. Chưa chạy lượt train 512 mới trên server; [A2_JupyterHub_Probe.ipynb](A2_JupyterHub_Probe.ipynb) chỉ chạy thử suy luận, không thay thế fine-tune.

Baseline ip2p_v1 đã train 2.000 steps trên 800 ảnh ở 256×256. Test 100 ảnh cho CLIP giảm 0,847809 → 0,843074; MAE giảm 0,166292 → 0,156616. Trên 94 ảnh không bị safety checker đánh dấu ở cả hai, cả hai proxy trung bình kém đi. Không dùng việc train lâu hơn như mặc định sẽ tốt hơn.

## Giai đoạn A — Probe rẻ, không train

- Chạy pretrained ở 512×512, 30 inference steps trên 20 validation ảnh, seed cố định.
- Sau khi A1 xong, nếu có ngân sách, thử text guidance {7,5;10} × image guidance {1,2;1,5;1,8}.
- Nếu đã chuyển checkpoint, chạy cùng cấu hình với fine-tuned U-Net 2.000 để đối chiếu. Nếu chưa chuyển, probe pretrained vẫn chạy được.
- Chấm trực quan đúng lệnh / giữ nội dung / artefact; metric CLIP-target và MAE chỉ là proxy. Giữ safety checker và đếm ảnh bị đánh dấu.
- Mở rộng validation lên toàn bộ 100 ảnh sau khi rút gọn còn 1–2 cấu hình. Không chọn guidance bằng test cũ đã xem.
- Đây là kế hoạch khảo sát; chưa khẳng định 512 hoặc guidance mới sẽ cải thiện ảnh.

## Giai đoạn B — Dữ liệu trước compute

Mục tiêu 5.000–10.000 cặp đã kiểm tra thay cho 800 mẫu. Một nguồn ứng viên là [HQ-Edit](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit), schema input_image/output_image/edit; cần adapter riêng và kiểm tra cách đóng gói/license (CC BY-NC 4.0 theo dataset card) trước khi dùng. **Chưa triển khai loader HQ-Edit** trong probe hiện tại, không tự tải toàn bộ dữ liệu lớn trên máy công ty.

Lọc cặp sai lệch quá nhiều ở vùng không cần sửa; nhóm ảnh nguồn trùng/gần trùng trước khi split. Giữ tập đánh giá mới độc lập cho lượt cải tiến vì test cũ đã xem và dataset nhỏ có nguy cơ overlap với pretrained training. Nếu thu hẹp miền (ví dụ màu sắc/thời tiết), công bố phạm vi và không khẳng định chỉnh sửa tổng quát.

## Giai đoạn C — Fine-tune có dừng sớm

Bắt đầu lại từ pretrained, không mặc định resume model 2.000 đã có dấu hiệu không cải thiện test. Cấu hình đề xuất để benchmark: 512×512, BF16, gradient checkpointing, batch 1–2 và accumulation giữ effective batch 16. Thử lr 5e-6 trước; chỉ thêm đối chứng 1e-5 nếu còn ngân sách.

Đo 100 optimizer steps trên server để ước tính chi phí trước khi chọn tổng steps. Chọn checkpoint bằng toàn bộ validation và ảnh minh họa định trước; không cố chạy đủ số epoch nếu chất lượng giảm. Chỉ mở test mới sau khi chốt cấu hình. Một seed là thử nghiệm bước đầu, không đủ chứng minh độ ổn định.

Ngân sách khi có riêng phiên A2 12 giờ: 1–2 giờ dữ liệu/probe/benchmark, 7–8 giờ train và validation, 1–2 giờ đánh giá/export. Các mốc là phân bổ dự kiến, không phải benchmark. Chừa thời gian lưu model trước khi server kết thúc. Trong phiên A1 hiện tại, không dành mặc định ngân sách train dài cho A2.

## Đầu ra cần có

Cấu hình và manifest dữ liệu; metric theo từng mẫu; ảnh đối chiếu cùng seed; checkpoint chọn bằng validation; thống kê ảnh safety flagged; thời gian/VRAM; báo cáo rõ phần tốt lên, kém đi và điều chưa kiểm chứng. Không cần Gradio hoặc web app: tất cả xem trực tiếp trong JupyterHub và export file nhỏ để tải về/đưa lên Drive.
