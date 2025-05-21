from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model (choose from yolov8n.pt, yolov8s.pt, etc.)
    model = YOLO("yolov8n.pt")

    model.train(data="train_cfg.yml",
                epochs=5000,
                imgsz=640,
                batch=16)
    
if __name__ == "__main__":
    main()