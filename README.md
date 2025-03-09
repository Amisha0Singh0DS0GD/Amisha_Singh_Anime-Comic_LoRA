# AnimeStyle Image Translation (LoRA Fine-Tuning)

## 📌 Project Overview
This project fine-tunes a pre-trained deep learning model using **LoRA (Low-Rank Adaptation)** to convert real images into anime-style images. The model is trained with optimized hyperparameters to enhance the quality of anime-style transformations.

## 📌 Project Structure
```
project-root/
│── anime-env/             # Virtual environment for dependencies
│── data/                  
│   ├── augmented_data/    # Augmented images
│   ├── original_data/     # Original dataset
│   ├── test_dataset/      # Test images
│   ├── dataset.md         # Dataset links stored on google drive
│── experiments/           
│   ├── model_checkpoints/ # Trained model checkpoints
│   ├── anime-style_training_log.csv  # Training logs
│   ├── experiments.md     # Links of checkpoints and training logs
│── fine-tuned-unet-lora/  # Fine-tuned model weights
│── notebooks/             # Jupyter notebooks for eda
│── Outputs/               # Generated anime-style images
│── scripts/               
│   ├── Model_run.py       # Main script to run the model
│── src/                   # Source code 
│── test/                  # Testing script
│── config.json            # Configuration file
│── Dockerfile             # Docker setup for reproducibility
│── requirements.txt       # Dependencies
```

## 📌 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-org/your-repo.git
cd your-repo
```
### **2️⃣ Create a Virtual Environment (Optional)**
```sh
python -m venv anime-env
source anime-env/bin/activate  # macOS/Linux
anime-env\Scripts\activate     # Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Train the Model**
Run the training script with LoRA fine-tuning:
```sh
python scripts/Model_run.py --config config.json
```

### **5️⃣ Generate Anime-Style Images**
Use the trained model for inference:
```sh
python scripts/Model_run.py --input path/to/image.jpg --output path/to/save.jpg
```

## 📌 Training Details
- **Model Used**: Fine-tuned UNet-based model with LoRA.
- **Dataset**: Augmented dataset of anime-style and real images.
- **Training Framework**: TensorFlow / PyTorch (specify accordingly).
- **Performance Metrics**: SSIM, PSNR, FID scores for image quality evaluation.

## 📌 Contribution Guidelines
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Added feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a Pull Request.

## 📌 Future Enhancements
- Optimize training for better anime-style generation.
- Implement real-time inference with optimized performance.
- Deploy as a web application.

## 📌 License
This project is licensed under the MIT License.

---
📌 **Author**: Amisha Singh

