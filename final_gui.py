from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model

# Load updated models
mnist_model = load_model('mnist_model_updated.h5')  # Use the updated model
devanagari_model = load_model('devanagari_model.h5')

image_folder = "img/"

# Initialize Tkinter
root = Tk()
root.resizable(0, 0)
root.title("Multi-Digit Recognizer")

lastx, lasty = None, None

cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=3)

# Clear the canvas
def clear_widget():
    cv.delete('all')

# Draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

cv.bind('<Button-1>', activate_event)

# Display recognized number in a new window
def show_number_window(number):
    number_window = Toplevel(root)
    number_window.title("Recognized Number")
    label = Label(number_window, text=f"Recognized Number: {number}", font=("Arial", 24))
    label.pack(padx=20, pady=20)

# Recognize digits based on selected model
def Recognize_Digit():
    filename = 'temp_digit.png'
    widget = cv

    # Capture canvas image
    x = root.winfo_rootx() + widget.winfo_rootx()
    y = root.winfo_rooty() + widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)

    # Read and process the image
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours and sort left-to-right
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    recognized_digits = []

    # Select model based on dropdown
    selected_model = model_var.get()
    model = mnist_model if selected_model == "MNIST" else devanagari_model

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore small noise
        if w < 5 or h < 5:
            continue

        # Extract and resize digit
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))

        # Pad to (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        # Ensure correct shape and normalize
        digit = padded_digit.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Predict digit
        pred = model.predict(digit)[0]
        final_pred = np.argmax(pred)

        recognized_digits.append(final_pred)

    # Combine digits into a proper number
    combined_number = int(''.join(map(str, recognized_digits))) if recognized_digits else 0

    print("Recognized Number:", combined_number)
    show_number_window(combined_number)

# Buttons
btn_save = Button(text='Recognize Digit', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Clear Widget', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

# Model selection dropdown
model_var = StringVar(root)
model_var.set("MNIST")  # Default model

model_menu = OptionMenu(root, model_var, "MNIST", "Devanagari")
model_menu.grid(row=2, column=2, pady=1, padx=1)

root.mainloop()
