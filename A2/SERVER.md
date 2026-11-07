# A2 trên server

Run `ip2p_v1` đã hoàn tất 2.000 optimizer steps, validation và test 100 ảnh. Xem [REPORT.md](REPORT.md); không cần train lại để có kết quả báo cáo này.

## 1. Git mang gì sang server?

Push source, notebook, REPORT.md và report_assets/. Thư mục A2/runs/ đã được gitignore: clone repo **không** mang theo checkpoint/model hoặc log đầy đủ. Không đưa optimizer nhiều GB vào Git.

- Chỉ suy luận/đánh giá: chuyển `runs/ip2p_v1/checkpoint-2000/unet/` (khoảng 3,44 GB) sang server, giữ nguyên tên folder. Phần model nền tải từ Hugging Face khi chạy.
- Muốn tiếp tục train: chuyển **toàn bộ** `checkpoint-2000/` (khoảng 10,32 GB gồm U-Net và optimizer, cùng scheduler/RNG nếu đã sync), `experiment.json` và `split_manifest.json`. Dùng checkpoint do chính nhóm tạo, không dùng file optimizer không rõ nguồn.
- Có thể dùng full pipeline ở `runs/ip2p_v1/` cho suy luận nếu chuyển đủ các thành phần, nhưng dung lượng lớn hơn chỉ mang U-Net.

Kiểm tra Google Drive đồng bộ xong trước khi chuyển. Hai file lớn của checkpoint 2000 ở lần kiểm tra: U-Net 3.438.213.624 bytes; optimizer 6.876.849.433 bytes. Giữ bản gốc trên Drive cho tới khi server nạp được.

## 2. Môi trường

Clone repo, tạo môi trường Python riêng. Cài PyTorch/torchvision tương thích driver GPU server theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/), sau đó:

```bash
cd SDH_HK261_CO5085/A2/instructpix2pix
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -v
python prepare_data.py --output /path/to/ip2p_data
```

Đối chiếu `split_manifest.json` mới với manifest trên Drive: seed, indices và source_image_hashes phải giống trước khi so sánh. Cập nhật các đường dẫn ví dụ bên dưới theo máy server.

## 3. Chạy đánh giá trên server

```bash
python compare.py --data /path/to/ip2p_data --split test --limit 100 --output ../runs/server_pretrained
python compare.py --data /path/to/ip2p_data --split test --limit 100 --unet ../runs/ip2p_v1/checkpoint-2000 --output ../runs/server_finetuned
```

Hai lệnh mặc định cùng seed 123, resolution 256, 20 inference steps. Thư mục output phải mới/trống. Đây là chạy lại đánh giá cùng test, không phải bằng chứng độc lập để chọn siêu tham số mới.

## 4. Tiếp tục train nếu cần thí nghiệm mới

Ví dụ tăng ngân sách lên 3.000 steps (khác lần thí nghiệm báo cáo):

```bash
python train_instruct_pix2pix.py \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --dataset_name /path/to/ip2p_data \
  --output_dir ../runs/ip2p_v1 \
  --resume_from_checkpoint checkpoint-2000 \
  --resolution 256 --center_crop \
  --train_batch_size 4 --gradient_accumulation_steps 4 \
  --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --conditioning_dropout_prob 0.05 --max_train_steps 3000 \
  --checkpointing_steps 100 --checkpoints_total_limit 2 \
  --dataloader_num_workers 2 --report_to tensorboard --seed 42
```

Chỉ dùng BF16 nếu GPU server hỗ trợ; nếu thay precision/batch, ghi rõ khác cấu hình gốc. Ngân sách 3.000 là tổng optimizer steps, không phải thêm 3.000. Chính sách giữ hai checkpoint sẽ dọn checkpoint cũ trong output_dir; giữ bản gốc trên Drive nếu muốn bảo toàn. Không điều chỉnh dựa trên test đã xem; chọn bằng validation hoặc một tập phát triển mới.
