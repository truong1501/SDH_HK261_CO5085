# A2 — InstructPix2Pix: chỉnh sửa ảnh theo chỉ dẫn

> A2.1 đã nộp DeepLabv3/Pascal VOC. Nhóm hiện chọn InstructPix2Pix theo yêu cầu của thành viên; chưa có xác nhận giảng viên đồng ý đổi đề tài. Nhóm sẽ trao đổi với giảng viên.

## Kết quả đã chạy

Run ip2p_v1 đã hoàn tất train 2.000 steps, validation và test 100 ảnh. Xem [báo cáo Markdown](REPORT.md), [bằng chứng nhỏ để commit](report_assets/) và [chuyển sang server](SERVER.md). Fine-tune chưa cải thiện đồng đều trên test; xem phân tích safety checker trong báo cáo.

## Chạy trên Colab A100

Mở **[A2_Colab_InstructPix2Pix.ipynb](A2_Colab_InstructPix2Pix.ipynb)** và chọn A100. Chạy lần lượt setup → dữ liệu → pilot 100 steps → validation → tiếp tục 2.000 steps → validation → test → demo. Đường dẫn Drive mặc định khớp thư mục của nhóm; sửa ROOT nếu cần.

- Checkpoint khởi đầu: `timbrooks/instruct-pix2pix`; fine-tune toàn bộ U-Net, đóng băng VAE/text encoder. Giữ nguyên conv 8 kênh đã học.
- Mặc định: 256×256, batch 4, accumulation 4, BF16 trên A100, lr 1e-5, constant scheduler, seed 42. Không flip ảnh vì lệnh có thể nói trái/phải.
- Có gradient checkpointing; lưu model/optimizer mỗi 100 steps và giữ hai bản gần nhất. Chuẩn bị hàng chục GB Drive trống. Resume bằng cùng RUN_ID và cấu hình; đặt RUN_ID mới cho thí nghiệm khác.
- Thời gian chưa benchmark: chạy pilot để đo, không cam kết số giờ. 2.000 steps là ngân sách khởi đầu, không đảm bảo chất lượng. Chọn checkpoint bằng validation trước khi mở test.
- Notebook không yêu cầu W&B hay upload model lên Hub. Có thể cần restart runtime sau cài phụ thuộc nếu phiên đã import các phiên bản khác.

## Dataset và hai biến thể

Dùng [fusing/instructpix2pix-1000-samples](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples): ảnh gốc, chỉ dẫn tiếng Anh và ảnh đích. Đây là tập nhỏ đa dạng; chưa lọc riêng thời tiết/thời điểm. Script kiểm tra schema và nhóm theo hash ảnh gốc để chia khoảng 80/10/10 theo nhóm; các bản chỉnh sửa cùng ảnh không đi qua nhiều split. Manifest/fingerprint được lưu để kiểm tra khi resume.

So sánh **pretrained** và **fine-tuned** trên cùng split, seed và thông số suy luận. Split chỉ tách dữ liệu của lần fine-tune; checkpoint pretrained có thể đã học tập dữ liệu gốc chứa những mẫu này. Muốn đánh giá khả năng tổng quát hóa độc lập cần thêm tập ảnh/chỉ dẫn riêng, chưa có trong bản này.

## Đánh giá

`compare.py` xuất ảnh input | reference | prediction, CSV từng mẫu và JSON tổng hợp. Metric: CLIP cosine giữa ảnh sinh và ảnh đích, paired pixel MAE, thời gian suy luận và số ảnh bị safety checker đánh dấu. Đây là proxy so với một ảnh tham chiếu, không phải độ chính xác làm theo lệnh; thay đổi hợp lệ có thể khác ảnh đích. Không dùng FID trên tập test nhỏ để kết luận chất lượng chỉnh sửa.

Kèm đánh giá con người 1–5 theo ba tiêu chí: đúng lệnh, bảo toàn nội dung không cần sửa, chất lượng/artefact. Chọn checkpoint bằng validation, sau đó đánh giá test một lần. Đã có kết quả thực nghiệm và báo cáo Markdown; chưa có PDF hoặc video báo cáo.

## Paper và code tham khảo

Tim Brooks, Aleksander Holynski, Alexei A. Efros. **InstructPix2Pix: Learning to Follow Image Editing Instructions**, CVPR 2023. [Paper](https://arxiv.org/abs/2211.09800) · [Scholar](https://scholar.google.com/scholar?q=InstructPix2Pix+Learning+to+Follow+Image+Editing+Instructions) · [Code tác giả](https://github.com/timothybrooks/instruct-pix2pix).

Ý tưởng: dữ liệu chỉnh ảnh tổng hợp; latent diffusion nhận ảnh nguồn và chỉ dẫn; conditioning dropout hỗ trợ guidance theo ảnh và văn bản. Thí nghiệm nhóm chỉ fine-tune checkpoint đã có, không tái tạo quy trình xây dựng dữ liệu và huấn luyện quy mô paper.

`instructpix2pix/train_instruct_pix2pix.py` lấy từ [Diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/instruct_pix2pix/train_instruct_pix2pix.py), giữ thông báo Apache-2.0. Thay đổi cục bộ: giữ U-Net 8 kênh; sao chép bias nếu chuyển từ 4 kênh; đọc DatasetDict save_to_disk và chỉ dùng split train. Phụ thuộc cố định trong `instructpix2pix/requirements.txt`. Dùng PyTorch/torchvision tương thích sẵn trên Colab, không tự cài lại CUDA.

## Mã trước đây

Hướng DiT/CelebA được giữ tại [README_DiT_previous.md](README_DiT_previous.md) và notebook DiT cũ. Các script `A2/train.py`, `sample.py`, `evaluate.py` thuộc hướng DiT; notebook InstructPix2Pix dùng riêng thư mục `instructpix2pix/`.

## Kiểm chứng cục bộ (07/11/2026)

- 3 unit tests đạt: split nhóm không rò rỉ, kiểm tra số nhóm tối thiểu, bảo toàn conv 8 kênh/chuyển 4 kênh đúng trọng số và bias.
- Script train import và CLI hoạt động với bộ phụ thuộc cố định. Đã chạy train 2 steps → lưu checkpoint → resume lên step 3 bằng mô hình Diffusers nhỏ ngẫu nhiên và dữ liệu tổng hợp trên CPU; model, optimizer và scheduler được nạp lại thành công.
- Accelerate ghi cảnh báo không khôi phục RNG trong môi trường CPU kiểm tra (PyTorch 2.15); resume không đảm bảo giống bit-for-bit với chạy liên tục. Cần kiểm tra thêm trên Colab/A100.
- Scripts và code cells notebook đã kiểm tra cú pháp; git diff --check đạt. Chưa tải checkpoint đầy đủ/dataset thật, chưa chạy compare CLIP hoặc fine-tune A100; chưa có kết quả chất lượng.
- Có thể chạy lại kiểm tra tích hợp: `python instructpix2pix/tests/smoke_cpu.py` sau khi cài requirements và PyTorch/torchvision.

## Hỗ trợ AI

OpenAI Codex hỗ trợ lựa chọn phương án, chỉnh script upstream, chuẩn bị split, đánh giá và notebook. Code trước có hỗ trợ Claude Code/Antigravity. Nhóm chịu trách nhiệm kiểm chứng dữ liệu, quyền sử dụng, số liệu và nội dung nộp. Chưa kiểm chứng fine-tune/checkpoint resume trên GPU A100 thật.

## Sửa setup cho Python 3.13 Colab

Đã cập nhật Transformers 4.57.1 và tokenizers 0.22.1 thay cho Transformers 4.44.2 (tokenizers cũ). Bộ phụ thuộc đã được resolver xác minh có wheel Linux/Python 3.13; train/checkpoint/resume mô hình nhỏ được kiểm tra lại trên Python 3.12. Cell setup hiện in đầy đủ log pip. Chưa thực thi notebook trên Colab Python 3.13.

## Lượt cải tiến sau baseline

Ưu tiên hoàn tất A1 trước. A2 chỉ probe ngắn tại [A2_JupyterHub_Probe.ipynb](A2_JupyterHub_Probe.ipynb); [kế hoạch cải tiến](IMPROVEMENT_PLAN.md) tách rõ probe, dữ liệu mới và fine-tune sau. Chưa có kết quả hoặc loader dữ liệu mới.
