import sys
from ultralytics import YOLO

def predict_image(image_path):
    # Đường dẫn đến file trọng số của mô hình đã huấn luyện
    model_path = 'train_results/classify/train/weights/best.pt'
    
    try:
        # Load mô hình YOLOv8
        model = YOLO(model_path)
        
        # Thực hiện dự đoán
        print(f"Đang phân tích ảnh: {image_path}...\n")
        results = model(image_path)
        
        # Trích xuất và in kết quả
        for result in results:
            # Lấy class có xác suất cao nhất (Top-1)
            top1_index = result.probs.top1
            predicted_class = result.names[top1_index]
            confidence = result.probs.top1conf.item()
            
            print("="*50)
            print("🏥 KẾT QUẢ SÀNG LỌC AI (CHEST X-RAY)")
            print("="*50)
            print(f"Ảnh đầu vào: {image_path}")
            print(f"Chẩn đoán:   {predicted_class}")
            print(f"Độ tin cậy:  {confidence * 100:.2f}%")
            print("="*50 + "\n")
            
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Lấy đường dẫn ảnh từ tham số dòng lệnh, nếu không có thì dùng ảnh mặc định
    if len(sys.argv) > 1:
        image_to_predict = sys.argv[1]
    else:
        # Sử dụng một trong các ảnh mẫu bạn vừa thêm
        image_to_predict = "COVID-1005.png"
        print("⚠️ Không có ảnh nào được truyền vào qua dòng lệnh.")
        print(f"Sử dụng ảnh mẫu mặc định: {image_to_predict}\n")
        
    predict_image(image_to_predict)
