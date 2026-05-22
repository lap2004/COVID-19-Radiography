import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Tải mô hình đã huấn luyện
model = YOLO('train_results/classify/train/weights/best.pt')

def predict(image):
    # Thực hiện dự đoán
    results = model(image)
    
    # Trích xuất kết quả
    result = results[0]
    
    # Tạo dictionary chứa xác suất của từng nhãn (Gradio yêu cầu format này)
    confidences = {model.names[i]: float(result.probs.data[i]) for i in range(len(model.names))}
    
    return confidences

# Cấu hình giao diện Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Tải lên ảnh X-quang phổi"),
    outputs=gr.Label(num_top_classes=4, label="Kết quả chẩn đoán"),
    title="🏥 Hệ thống AI Sàng lọc Bệnh lý Hô hấp (Chest X-Ray)",
    description="Tải lên một tấm ảnh X-quang ngực (Chest X-Ray) để AI chẩn đoán. Hệ thống có thể phân loại 4 trường hợp: COVID-19, Viêm phổi thường (Viral Pneumonia), Mờ phổi (Lung Opacity) và Phổi bình thường (Normal).",
    examples=[
        ["COVID-1005.png"],
        ["Normal-10000.png"],
        ["Lung_Opacity-1004.png"],
        ["Viral Pneumonia-100.png"]
    ],
    theme="huggingface"
)

if __name__ == "__main__":
    # Khởi chạy ứng dụng Web (share=True để tạo link public cho người khác dùng thử)
    interface.launch(share=True)
