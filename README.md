# Image Forgery Detection Web App

A deep learning-powered web application to detect **Copy-Move Image Forgery** using a custom-trained convolutional neural network (CNN). Built with **TensorFlow/Keras**, trained via Jupyter Notebook, and deployed using **Flask**.
![image](https://github.com/user-attachments/assets/66a30751-ca39-4a8e-9c63-71c7f376de8e)

---

## What is Image Forgery?

Image forgery refers to manipulating a digital image to alter its content, often to deceive viewers. One common method is **Copy-Move Forgery**, where a part of an image is copied and pasted to conceal or duplicate elements.

---

## How to use

Upload an image via the web app interface, and the model will analyze it to detect possible forgery regions.

<p align="center">
  <img src="https://github.com/your-username/image-forgery-detection/assets/demo.gif" width="600"/>
</p>

---

##  Project Structure
Train the model and save into artifacts folder
Run all cells to train the model.
jupyter notebook Research.ipynb

Activate the virtual env
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Run the web app
python app.py
