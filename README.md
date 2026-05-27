# Machine Learning-based Anomaly Detection for Securing LMS against Credential Stuffing Attacks

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange)

## Giới thiệu (Introduction)
Dự án này nghiên cứu và phát triển một hệ thống phát hiện điểm bất thường dựa trên học máy (Machine Learning) nhằm bảo vệ các Hệ thống Quản lý Học tập (Learning Management Systems - LMS) khỏi các cuộc tấn công nhồi nhét thông tin xác thực (Credential Stuffing). 

Hệ thống không chỉ tập trung vào việc phát hiện các kỹ thuật tấn công tinh vi (như low-and-slow attacks) mà còn tích hợp cơ chế phòng thủ cho mô hình AI và tự động hóa quy trình phản hồi sự cố.

Đây là Đồ án Tốt nghiệp thuộc chương trình Công nghệ Thông tin Ứng dụng tại **Khoa Quốc tế - Đại học Quốc gia Hà Nội (VNU-IS)**. Xin gửi lời cảm ơn chân thành đến **TS. Nguyễn Văn Tánh** đã tận tình hướng dẫn và hỗ trợ hoàn thành dự án này.

## Các tính năng chính (Key Features)
- **Mô phỏng tấn công & Dữ liệu:** Kịch bản giả lập lưu lượng truy cập bình thường và các cuộc tấn công Credential Stuffing nhắm vào hệ thống LMS (Moodle).
- **Phát hiện bất thường (Anomaly Detection):** Pipeline phân tích hành vi đăng nhập để nhận diện sớm dấu hiệu xâm nhập.
- **Bảo vệ toàn vẹn AI (Anti-Data Poisoning):** Ứng dụng thuật toán phòng thủ nhằm bảo vệ mô hình ML trước các kỹ thuật đầu độc dữ liệu huấn luyện.
- **Tích hợp SOAR:** Mô phỏng tự động hóa quy trình điều phối và phản hồi bảo mật (Security Orchestration, Automation, and Response) để xử lý các cảnh báo an toàn thông tin theo thời gian thực.

##  Cấu trúc Repository (Repository Structure)
- `charts/` : Các biểu đồ phân tích và đánh giá hiệu suất của mô hình.
- `deploy_models/` : Thư mục chứa các mô hình học máy đã được huấn luyện, sẵn sàng để triển khai.
- `templates/` : Tệp giao diện HTML phục vụ cho Web App mô phỏng hệ thống SOAR.
- `anomaly_detection_pipeline.py` & `.ipynb` : Pipeline chính để tiền xử lý dữ liệu và huấn luyện mô hình phát hiện xâm nhập.
- `anti_data_poisoning_defense.ipynb` : Giải pháp bảo vệ thuật toán học máy khỏi rủi ro Data Poisoning.
- `low_and_slow_attack.py` : Kịch bản tự động mô phỏng hình thức tấn công Credential Stuffing tần suất thấp nhằm qua mặt các hệ thống giám sát truyền thống.
- `normal_traffic.py` & `generate_combos.py` : Script sinh tập dữ liệu hành vi người dùng hợp lệ và danh sách thông tin xác thực.
- `soar_simulation.py` & `soar_web_app.py` : Hệ thống mô phỏng cảnh báo và cung cấp giao diện quản trị sự cố bảo mật tự động.
- `moodle_students.csv`, `leaked_combos.txt`, `lms_training_dataset_final.csv`: Dữ liệu thô và tập dữ liệu mẫu được sử dụng để huấn luyện và kiểm thử.

##  Hướng dẫn sử dụng (Getting Started)
1. Cài đặt các thư viện phụ thuộc (Dependencies).
2. Chạy các kịch bản sinh dữ liệu giả lập lưu lượng và tấn công.
3. Thực thi `anomaly_detection_pipeline.ipynb` để bắt đầu quá trình huấn luyện mô hình.
4. Khởi chạy `soar_web_app.py` để mở giao diện quản lý sự cố và theo dõi log event.

##  Tác giả (Author)
- Tanie (Nguyễn Thảo Anh)
- Lĩnh vực nghiên cứu: Application Security & AI Security
