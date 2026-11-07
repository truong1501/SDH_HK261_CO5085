# A2: U-Net DDPM và DiT thu nhỏ trên CelebA

> **Trạng thái đổi đề tài:** A2.1 đã nộp với DeepLabv3 / Pascal VOC 2012. Theo yêu cầu của nhóm, code hiện chuyển sang tạo sinh ảnh với DiT / CelebA. Chưa có xác nhận giảng viên đồng ý; nhóm sẽ liên hệ giảng viên. Đây không phải nội dung A2.1 đã được phê duyệt.

## Chủ đề

Tạo sinh hình ảnh (Generative Image Modeling): Huấn luyện mô hình xác suất khuếch tán (Diffusion Models) để sinh ảnh chân dung từ nhiễu Gaussian ngẫu nhiên.

## Paper

William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers* (DiT), **ICCV 2023 (Oral)**. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).

Ngày 07/11/2026 đã truy cập Google Scholar và xác minh bài DiT theo tên/tác giả. Danh sách gợi ý trước đó dựa trên kiến thức AI, không phải kết quả tìm Scholar. [Tìm trên Google Scholar](https://scholar.google.com/scholar?q=Scalable+Diffusion+Models+with+Transformers).

File PDF bài báo lưu tại: `Exercises/SDH_HK261_CO5085/A2/papers/DiT_ICCV2023_Peebles_Xie.pdf`.

## Dữ liệu

Tập dữ liệu **CelebA (Large-scale CelebFaces Attributes Dataset)** gồm 202.599 ảnh khuôn mặt của 10.177 người, mỗi ảnh có 40 thuộc tính nhị phân và 5 điểm mốc khuôn mặt. Pipeline hiện chỉ dùng ảnh, không dùng nhãn thuộc tính. [Nguồn và tải chính thức](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html); chỉ dùng cho nghiên cứu phi thương mại, không đưa dữ liệu lên GitHub.

- Tập chuẩn công khai được khuyến nghị trong đề bài A2 (mục 3.2).
- Sử dụng tập ảnh khuôn mặt chuẩn hóa và căn chỉnh (Aligned & Cropped), resize về độ phân giải $64 \times 64$ (hoặc $128 \times 128$), chuẩn hóa giá trị pixel về dải $[-1, 1]$.
- Ưu tiên split chính thức qua `list_eval_partition.txt`: 0=train, 1=validation, 2=test. Nếu thiếu file, dùng split 80/10/10 theo seed 42; phải nêu rõ trong báo cáo. Chỉ train có flip ngẫu nhiên. Validation chọn checkpoint, test dùng tính FID; không lấy test để chọn mô hình.

## Metric

- **FID (Fréchet Inception Distance)**: Metric chính đánh giá độ chân thực và tính đa dạng của phân phối ảnh sinh ra so với ảnh thực tế trong tập dữ liệu (tính qua thư viện `pytorch-fid` / `torch-fidelity`).
- **IS**: Chưa cài đặt; ưu tiên FID cho thí nghiệm hiện tại.
- **Tốc độ sinh ảnh:** script sample báo images/s, gồm cả chuyển ảnh và lưu PNG; ghi cùng thiết bị, batch size, số bước khi so sánh.
- **Độ phức tạp mô hình**: Số lượng tham số (Parameters), kích thước checkpoint và thời gian huấn luyện mỗi epoch.

## Kế hoạch pipeline

1. **Khảo sát và chuẩn bị dữ liệu (EDA & DataLoader):**
   - Tải và tiền xử lý tập dữ liệu CelebA, kiểm tra phân bố độ sáng, kích thước.
   - Xây dựng PyTorch Dataset và DataLoader tối ưu với pipeline augmentation (Random Horizontal Flip) và chuẩn hóa $[-1, 1]$.
2. **Xây dựng quy trình khuếch tán (Diffusion Process):**
   - Thiết lập Forward process: Thêm nhiễu Gaussian theo lịch trình phương sai (Cosine / Linear variance schedule) với $T = 1000$ bước.
   - Hàm mục tiêu huấn luyện: Dự đoán nhiễu $\epsilon$ với hàm mất mát MSE Loss $\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$.
3. **So sánh hai kiến trúc khử nhiễu (Denoising Backbones):**
   - **Biến thể 1 (CNN truyền thống):** Kiến trúc U-Net kinh điển (chuẩn DDPM của Ho et al., NeurIPS 2020) với Convolutional ResNet blocks và Multi-head Self-Attention tại bottleneck.
   - **Biến thể 2 (Transformer hiện đại - DiT):** Kiến trúc Diffusion Transformer (Peebles & Xie, ICCV 2023) với kỹ thuật Patchify, chuỗi DiT Transformer blocks với cơ chế điều kiện hóa tham số hóa thích nghi (adaLN-Zero).
4. **Huấn luyện & Đánh giá thực nghiệm:**
   - Huấn luyện cả hai mô hình trên cùng một tập dữ liệu CelebA và cùng lịch trình thêm nhiễu.
   - So sánh đường cong suy giảm hàm mất mát (loss curves) theo epoch.
   - Trực quan hóa tiến trình khử nhiễu từ lúc là nhiễu trắng thuần túy $t=T$ đến khi hình thành khuôn mặt rõ nét $t=0$.
   - Đo đạc định lượng FID score, số tham số và thời gian lấy mẫu.
5. **Phân tích bài báo (Paper Analysis) & Báo cáo:**
   - Phân tích chi tiết kiến trúc DiT: cơ chế adaLN-Zero, quy luật mở rộng (Scaling Laws), bằng chứng scaling trong paper; không mặc định DiT thu nhỏ của nhóm tốt hơn U-Net.
   - Chỉ ra các hạn chế của DiT (chi phí tính toán cơ chế self-attention $O(N^2)$ khi ảnh kích thước lớn nếu không qua Latent space).
   - Đối chiếu kết quả thực nghiệm của nhóm với công bố gốc trong bài báo.

## Phạm vi tái hiện paper

Paper gốc dùng latent diffusion trên ImageNet có điều kiện lớp; DiT lớn nhất đạt FID 2,27 trên ImageNet 256×256. Bản cài đặt này dùng ảnh RGB CelebA 64×64, không VAE, không nhãn lớp, không classifier-free guidance, không học variance; positional embedding học được. Giữ ý tưởng patchify + attention + adaLN-Zero, không tái hiện nguyên vẹn paper. Cấu hình hidden=384, depth=8, heads=6 là cấu hình thu nhỏ riêng, không gọi là DiT-S chuẩn. U-Net cũng là baseline giản lược.

Đối chiếu thí nghiệm: train cùng split, seed, số epoch, resolution, lịch nhiễu và learning rate; báo thêm số tham số vì hai mô hình không cùng kích thước. Loss dự đoán nhiễu không thay thế FID. Chưa có kết quả CelebA để kết luận mô hình nào tốt hơn.

## Chạy pipeline

Mở `A2_Colab_DiT_Pipeline.ipynb` trong Colab và chọn GPU. Tải `img_align_celeba.zip` cùng `list_eval_partition.txt` từ trang CelebA chính thức lên Drive; notebook giải nén ảnh vào `/content` để giảm thời gian đọc file. Batch mặc định 16; nếu thiếu VRAM, giảm batch cho cả hai mô hình.

Từ thư mục `A2`:

```bash
pip install -r requirements.txt
python train.py --model unet --data_dir /path/img_align_celeba --partition_file /path/list_eval_partition.txt --save_dir runs/unet --sample_dir runs/unet/samples --batch_size 16
python train.py --model dit --data_dir /path/img_align_celeba --partition_file /path/list_eval_partition.txt --save_dir runs/dit --sample_dir runs/dit/samples --batch_size 16
python sample.py --ckpt runs/dit/dit_best.pt --num_images 4 --batch_size 4 --save_steps --output runs/dit/denoising.png
python sample.py --ckpt runs/dit/dit_best.pt --num_images 5000 --batch_size 4 --output_dir runs/dit/generated
python evaluate.py --data_dir /path/img_align_celeba --partition_file /path/list_eval_partition.txt --generated_dir runs/dit/generated --real_dir runs/dit/real_test --num_images 5000 --output runs/dit/fid.json
python -m unittest discover -s tests -v
```

Lặp lại sample/evaluate với checkpoint U-Net. Đổi `save_dir` cho lần train mới để giữ kết quả. Không hỗ trợ resume trong bản hiện tại; checkpoint có optimizer/scaler cho bước mở rộng sau. Checkpoint cũ thiếu config cần train lại. Có thể dùng `--no-fp16`, `--num_workers 0`, `--max_samples 100` để thử nhanh (giới hạn áp dụng riêng cho mỗi split). `max_samples` nhỏ chỉ kiểm tra code, không dùng kết luận chất lượng.

Đầu ra: CSV gồm train/val MSE và thời gian epoch; checkpoint best theo val MSE và last; ảnh mẫu; `fid.json` chứa metric và cấu hình. FID dùng feature Inception 2048 chiều của `pytorch-fid`, 5.000 ảnh thật test và 5.000 ảnh sinh riêng lẻ, resize giống nhau. Đây là FID với 5k mẫu của nhóm, không so trực tiếp với FID 50k/khác dataset của paper. Lần chạy FID đầu cần mạng tải trọng số Inception. Ảnh ghép chỉ để xem, không dùng làm input FID.

## Trạng thái kiểm chứng

Ngày 07/11/2026: 3 unit tests đạt trên Python 3.12.14, PyTorch 2.15.0+cpu / torchvision 0.29.2+cpu. Đã chạy CLI end-to-end cho cả U-Net (27.198.467 tham số) và DiT (21.860.784 tham số): một epoch, train/val, checkpoint best/last, sinh 3 PNG theo batch 2 và xuất tiến trình khử nhiễu; dùng ảnh tổng hợp 8×8, lịch linear 2 bước để xác minh sample dùng đúng config checkpoint. Mã Python và toàn bộ code cells notebook đã parse thành công. Chưa chạy Colab/GPU hoặc tính FID thật.

Code cần được huấn luyện trên CelebA để có số liệu A2.2. Kiểm tra kỹ thuật với ảnh tổng hợp không chứng minh mô hình đã học sinh khuôn mặt. Chưa có FID thực nghiệm, báo cáo PDF hoặc video demo trong phần triển khai này.

## Hỗ trợ AI

Phương án chủ đề, paper, kế hoạch và khung code A2 được soạn với hỗ trợ của trợ lý AI (Claude Code, Antigravity, OpenAI Codex). Codex hỗ trợ rà soát, sửa pipeline train/validation/checkpoint/FID, đồng bộ notebook và kiểm tra bằng dữ liệu tổng hợp. Nhóm chịu trách nhiệm kiểm chứng dữ liệu, kết quả và nội dung nộp.
