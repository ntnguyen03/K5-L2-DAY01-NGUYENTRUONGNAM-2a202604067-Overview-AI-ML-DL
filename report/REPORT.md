# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    - Record này nói về cả bức ảnh thay vì chỉ một vật thể riêng vì mô hình được sử dụng trong cell code này là 1 mô hình phân loại, mô hình này sẽ phân loại từng bức ảnh thay vì từng object trong ảnh
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    - Được xác định bởi bộ dữ liệu và taxonomy dùng để huấn luyện mô hình ở đây là bộ dữ liệu ImageNet-1K
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    - Nếu chỉ ghi mỗi tên lớp thì có thể bị nhầm lẫn vì tên lớp có thể trùng nhau, ID giúp cho máy tính xử lý dễ dàng hơn (thuận tiện cho việc đối chiếu với các tham số), giúp tái sử dụng hiệu quả với taxomony vì nhiều dataset khác nhau có thể bị trùng tên gọi cho 1 class
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    - Guideline cần quy định rõ các tiêu chí để chọn một lớp để thống nhất để annotator chọn đúng lớp (ví dụ: chủ thể chính, mức độ nổi bật, độ rộng, ngữ cảnh,...)
6. Vì sao không được lấy `score` của model làm ground truth?
    - Vì score chỉ thể hiện mức độ đúng về một dự đoán, chứ không chứng minh được dự đoán đó có đúng hay không so với thực tế 
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    - bbox sử dụng định dạng xyxy (xmin, ymin, xmax, ymax): góc trên trái là xmin, ymin và góc dưới phải là xmax, ymax
    - rộng 113.58 pixel(bbox_width) và cao 279.68 pixel (bbox_height)
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    - Hai ngưỡng trên ảnh kitchen: 0.20 nhiều prediction hơn (17 vật thể) 0.60 ít prediction hơn (6 vật thể)
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    - Người reviewer phải check nhiều box hơn, kể cả những box nhiễu vì độ tin cậy thấp bị lẫn vào, lượng công việc tăng
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    - Mỗi bbox chỉ bao quanh một đối tượng cần gán nhãn, hộp phải bao sát toàn bộ phần đối tượng nhìn thấy được, không lấy thừa quá nhiều vùng nền xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Một vật thể bị che khuất là person phía bên góc trái bức ảnh. Đưa quy tắc xử lý đối tượng bị cắt vào guideline, trường hợp không đủ nhận diện thì escalation

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-003 ; `class_name`=bowl ; `score`=0.698698 ; số điểm polygon=`polygon_point_count`=70 ; 3–5 điểm đầu của `polygon_xy`=[84.0, 289.0], [83.0, 290.0], [81.0, 290.0],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - Mô tả đường biên thực tế của vật thể, phân biệt vùng vật thể với vùng nền, Hỗ trợ xác định hình dạng và diện tích vật thể chính xác hơn
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - Dùng để xác định một đối tượng cụ thể trong ảnh, không phải class id vì nó không đại diện cho loại vật thể, không phải tracking id vì không theo dõi vật thể xuyên suốt các frame video
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    - Đường biên polygon phải bám sát đường viền thực tế của đối tượng, chỉ bao gồm phần vật thể nhìn thấy được, không bao gồm nền hoặc vật thể khác.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Chỗ 3 cái bát chạm nhau trong ảnh kitchen. Dùng guidline nếu đây là tình huống phổ biến và có thể đưa ra quy tắc thống nhất. Escalation nếu không xác định ranh giới giữa 2 vật thể
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một ảnh tương ứng đúng một nhãn lớp (class_name), thuộc taxonomy(ImageNet-1K) xác định trước  |Ảnh traffic có nhiều phương tiện nhưng prediction top-1 là cab, score 0.510915. Khó quyết định lớp đại diện cho toàn ảnh|Xác định chủ đề chính theo guideline và gán duy nhất một nhãn cho ảnh|Kiểm tra nhãn có phù hợp nội dung ảnh, đúng taxonomy và nhất quán giữa các annotator không|
| Phát hiện vật thể |Mỗi vật thể có một nhãn lớp và một bounding box (class_name, bbox_xyxy)|Ảnh kitchen có nhiều vật thể nằm gần nhau. JSON có person score 0.912625, bbox [385.33, 69.24, 498.92, 348.92]Có người bị cắt ở mép trái ảnh, cần chú ý khi gán nhãn|Xác định từng vật thể, gán class và vẽ box bao sát phần vật thể nhìn thấy|Kiểm tra class, vị trí box, box có bị thiếu, thừa, lệch hoặc trùng lặp không|
| Instance segmentation |Mỗi đối tượng có một nhãn lớp và một mask/polygon riêng biệt (class_name, polygon_xy, instance_id)|instance_id=kitchen-003, class_name=bowl, score 0.698698, polygon gồm 70 điểm. Các bát và dụng cụ nằm gần nhau khiến ranh giới giữa các vật thể khó xác định|Vẽ polygon riêng cho từng instance, bám sát đường biên nhìn thấy được, không suy đoán phần bị che|Kiểm tra mask có đúng vật thể, đường biên có sát, có lấn sang vật khác hoặc bỏ sót vùng vật thể không|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    - Chỉ dùng ba ảnh COCO công khai có sẵn — không tải ảnh cá nhân hay dữ liệu nội bộ, không huấn luyện hoặc tinh chỉnh mô hình trong bài này. Không tự đổi checkpoint, threshold hoặc mã nguồn trừ khi GV cho phép
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    - Tôi sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu, sau đó báo cáo trực tiếp cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn xử lý

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
