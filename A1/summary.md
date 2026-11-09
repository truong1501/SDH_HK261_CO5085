# Kết quả A1

| Run | Mô hình | Chế độ | LR backbone | Seeds | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | Trainable | Thời gian train (phút) | VRAM đỉnh (GB) | Ảnh/s | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | resnet50 | From-scratch | 0.0005 | 1 | 82.48 | 95.85 | 82.32 | 23.71M | 112.5 | 11.8 | 6287 | NVIDIA A100-SXM4-40GB |
| R2 | resnet50 | Freeze toàn bộ | — | 3 | 71.06 ± 0.05 | 91.56 ± 0.02 | 70.86 ± 0.04 | 0.21M | 1.8 | 2.6 | 7006 | NVIDIA A100-SXM4-80GB |
| R3 | resnet50 | Freeze một phần | 0.0003 | 3 | 81.77 ± 0.12 | 95.90 ± 0.03 | 81.69 ± 0.13 | 15.17M | 16.5 | 1.8 | 6294 | NVIDIA A100-SXM4-40GB |
| R4 | resnet50 | Full finetune | 0.0003 | 3 | 85.65 ± 0.01 | 97.11 ± 0.07 | 85.57 ± 0.04 | 23.71M | 17.1 | 11.8 | 6286 | NVIDIA A100-SXM4-40GB |
| R5 | deit_s | From-scratch | 0.0005 | 1 | 67.13 | 88.52 | 66.60 | 21.70M | 111.0 | 9.3 | 9190 | NVIDIA A100-SXM4-40GB |
| R6 | deit_s | Freeze toàn bộ | — | 3 | 72.79 ± 0.09 | 92.35 ± 0.04 | 72.65 ± 0.11 | 0.04M | 1.7 | 1.0 | 9374 | NVIDIA A100-SXM4-80GB |
| R7 | deit_s | Freeze một phần | 0.0001 | 3 | 87.25 ± 0.14 | 97.58 ± 0.02 | 87.22 ± 0.12 | 10.69M | 16.5 | 5.0 | 9189 | NVIDIA A100-SXM4-40GB |
| R8 | deit_s | Full finetune | 0.0001 | 3 | 88.62 ± 0.10 | 97.99 ± 0.04 | 88.60 ± 0.08 | 21.70M | 16.9 | 9.3 | 9329 | NVIDIA A100-SXM4-80GB |

## Khoảng cách pretrained - from-scratch

| model    |   scratch |   best_pretrained |   gap |
|:---------|----------:|------------------:|------:|
| deit_s   |     67.13 |             88.62 | 21.49 |
| resnet50 |     82.48 |             85.65 |  3.16 |

## 10 cặp lớp nhầm nhiều nhất


### deit_s (R8, seed 42, top-1 88.61%)

| Nhãn thật | Bị đoán thành | Số ảnh |
|---|---|---|
| steak | filet_mignon | 44 |
| filet_mignon | steak | 39 |
| chocolate_mousse | chocolate_cake | 21 |
| prime_rib | steak | 21 |
| chocolate_cake | chocolate_mousse | 18 |
| tuna_tartare | beef_tartare | 16 |
| pork_chop | filet_mignon | 16 |
| dumplings | gyoza | 16 |
| beef_tartare | tuna_tartare | 16 |
| apple_pie | bread_pudding | 16 |

### resnet50 (R4, seed 43, top-1 85.64%)

| Nhãn thật | Bị đoán thành | Số ảnh |
|---|---|---|
| steak | filet_mignon | 46 |
| chocolate_cake | chocolate_mousse | 29 |
| filet_mignon | steak | 26 |
| pork_chop | filet_mignon | 24 |
| steak | prime_rib | 24 |
| tuna_tartare | beef_tartare | 23 |
| pork_chop | steak | 20 |
| chocolate_mousse | chocolate_cake | 17 |
| steak | pork_chop | 17 |
| prime_rib | steak | 16 |

## Dò LR backbone (full finetune, seed 42, validation top-1 %)

| Mô hình | 5e-05 | 0.0001 | 0.0003 | Chọn |
|---|---|---|---|---|
| ResNet-50 (CNN) | 75.12 | 77.91 | 81.35 | 0.0003 |
| DeiT-S (ViT) | 83.51 | 84.63 | 84.58 | 0.0001 |

## Đồng thuận CNN vs ViT (seed 42, McNemar)

| Chế độ | Cả hai đúng | Chỉ resnet50 đúng | Chỉ deit_s đúng | Cả hai sai | Sai cùng nhãn | χ² | p |
|---|---|---|---|---|---|---|---|
| From-scratch | 63.94% | 18.54% | 3.19% | 14.32% | 5.45% | 2736.1 | 0 |
| Freeze toàn bộ | 62.07% | 9.01% | 10.62% | 18.30% | 7.06% | 33.1 | 8.8e-09 |
| Freeze một phần | 78.23% | 3.61% | 9.02% | 9.14% | 4.78% | 585.1 | 2.9e-129 |
| Full finetune | 82.06% | 3.60% | 6.55% | 7.79% | 4.36% | 216.0 | 6.8e-49 |

## Lớp dễ/khó (run tốt nhất theo val)

- **ResNet-50 (CNN)** khó nhất: steak (51%), pork_chop (59%), apple_pie (62%), bread_pudding (67%), ceviche (69%); dễ nhất: edamame (100%), macarons (97%), cannoli (96%), hot_and_sour_soup (96%), spaghetti_bolognese (96%)
- **DeiT-S (ViT)** khó nhất: steak (62%), pork_chop (65%), filet_mignon (70%), bread_pudding (73%), chocolate_mousse (75%); dễ nhất: edamame (100%), spaghetti_bolognese (98%), pad_thai (98%), hot_and_sour_soup (97%), pho (97%)