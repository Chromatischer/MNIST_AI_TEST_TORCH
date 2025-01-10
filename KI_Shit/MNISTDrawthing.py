from tkinter import *

import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image, ImageGrab

import CNN_Network
import NeuralNetWithBatchNorm

EXTENDED_DEBUG_INFO = False
CLEAR_CANVAS_AFTER_PREDICTION = True
POPUP_KYS_TIME = 30

# Define the class for your Tkinter-based drawing application
class DigitRecognizerApp:
    def __init__(self, model_path, window_title="Digit Recognizer"):
        self.window = Tk()
        self.window.title(window_title)
        self.window.geometry("310x450")
        self.window.resizable(False, False)


        self.current_popup = None # Destroying old popups using this

        # Canvas to draw the digit
        self.canvas = Canvas(self.window, width=308, height=308, bg="white") # some multiple of 28x28
        self.canvas.pack()

        # Buttons for clearing the canvas and predicting the digit
        self.clear_button = Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=LEFT, padx=10)

        self.predict_button = Button(self.window, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=RIGHT, padx=10)

        self.close_button = Button(self.window, text="Close", command=self.close_app) # doesn't show up...
        self.close_button.pack(side=RIGHT, padx=100)

        # Load the trained PyTorch model
        print("Loading model...")
        model = CNN_Network.CNNForMNIST().to("cpu")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()  # Set the model to evaluation mode
        self.model = model
        print("Model loaded successfully. From: " + model_path)

        # Add event bindings for drawing
        self.canvas.bind("<B1-Motion>", self.draw)

        # Placeholder for coordinates of the drawn strokes
        self.x = self.y = None

    def draw(self, event):
        """Draw a line on the canvas where the mouse moves."""
        if self.x and self.y:
            self.canvas.create_line((self.x, self.y, event.x, event.y), width=14, fill="black", capstyle=ROUND, smooth=TRUE)
        self.x, self.y = event.x, event.y

    def clear_canvas(self):
        """Clear the canvas for a new drawing."""
        self.canvas.delete("all")
        self.x = self.y = None

    @DeprecationWarning
    def capture_image(self):
        """Capture the drawing and save it as an image.
        DEPRECATED, USE: capture_image_new!
        """
        x = self.window.winfo_rootx() + self.canvas.winfo_x()
        y = self.window.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
        return img

    def capture_image_new(self):
        self.canvas.postscript(file="tmp.ps", colormode='gray')
        img = Image.open("tmp.ps")
        img = img.convert("L")
        return img

    def close_app(self):
        """Close the Tkinter application."""
        self.window.destroy()
        exit(0)

    def predict_digit(self):
        """Capture the drawing, process it, and use the model to predict the digit."""
        print("Predicting digit...")
        # Save the canvas content as an image
        img = self.capture_image_new()
        print("Image captured")

        # Preprocess the image to match the model input
        img = img.resize((28, 28), Image.Resampling.BICUBIC) # Resize to 28x28 pixels
        if EXTENDED_DEBUG_INFO: img.show(title="Resized Image")

        # Convert to numpy array
        img_array = np.asarray(img)
        if EXTENDED_DEBUG_INFO: print(img_array)

        print("Image array created", img_array.shape)

        img = Image.fromarray(img_array, mode="L")
        if EXTENDED_DEBUG_INFO: img.show(title="Image Array")

        #invert colors of the image
        img_array =  ((1 - img_array / 255.0) * 2) - 1 # img_array / 255.0 -> 0 to 1, * 2 -> 0 to 2, X - 1 -> -1 to 1
        if EXTENDED_DEBUG_INFO: NeuralNetWithBatchNorm.printCleanNumpyArray(img_array)

        #show the image after processing
        another_array = ((img_array + 1) / 2) * 255
        if EXTENDED_DEBUG_INFO: NeuralNetWithBatchNorm.printCleanNumpyArray(another_array)

        img = Image.fromarray(another_array)
        if EXTENDED_DEBUG_INFO: img.show(title="After Inversion")

        print("Image array inverted")

        img_tensor = torch.tensor(img_array, dtype=torch.float32)  # Normalize and add batch/channel dims

        img_tensor = img_tensor.unsqueeze(0) # Converting to dimension [0, 28, 28] (Channel Dimension)

        img_tensor = img_tensor.unsqueeze(0) # Converting into 4D [0, 0, 28, 28] (Batch Dimension)

        print("Converted Image Dimension!")
        if EXTENDED_DEBUG_INFO: print("Converted Image Shape: ", img_tensor.shape)

        # Use the model to predict
        with torch.no_grad():
            prediction = self.model(img_tensor)

            probabilities = f.softmax(prediction[0], dim=0)

            if EXTENDED_DEBUG_INFO:
                print(f"Raw logits: {prediction}")
                print(f"Probabilities. {probabilities}")

            # Better prediction display in command line:
            for i in range(10):
                print(f"{i}: {probabilities[i].item() * 100:.2f}% logit: ({prediction[0][i].item():.2f})")

            predicted_digit = torch.argmax(prediction).item()

        # Show the prediction
        self.show_prediction(predicted_digit, probabilities[predicted_digit].item() * 100)

        # Clear the canvas
        if CLEAR_CANVAS_AFTER_PREDICTION: self.clear_canvas()

    def show_prediction(self, digit, accuracy):
        """Display the predicted digit in a popup window."""

        # Ensure that there ist only one single popup on screen at a time!
        if self.current_popup is not None:
            self.current_popup.destroy()
            self.current_popup = None

        self.current_popup = Toplevel(self.window)
        self.current_popup.title("Prediction")
        label = Label(self.current_popup, text=f"Predicted Digit: {digit}", font=("Arial", 20))
        label.pack(pady=20)

        if accuracy < 80:
            label = Label(self.current_popup, text=f"Accuracy: {accuracy:.2f}% (Unsure)", font=("Arial", 15), fg="red")
        else:
            label = Label(self.current_popup, text=f"Accuracy: {accuracy:.2f}% (Sure)", font=("Arial", 15), fg="green")

        label.pack(pady=20)

        button = Button(self.current_popup, text="Close", command=self.current_popup.destroy)
        button.pack(pady=10)

        self.current_popup.after(POPUP_KYS_TIME * 1000, self._auto_destroy_popup) # Auto destruct the window after some amount of time!

    def _auto_destroy_popup(self):
        """Function to automatically destroy a popup after some amount of time"""
        if self.current_popup is not None:
            self.current_popup.destroy()
            self.current_popup = None
            print("Auto destructed window!")

    def run(self):
        """Run the Tkinter main loop."""
        self.window.mainloop()

# Main script
if __name__ == "__main__":
    print("Starting the Application")
    # Replace 'path_to_model.pth' with the actual path to your trained model file
    # both model 21 and model 22 use CNNs! 22 has been trained for more Epochs, untested though!
    app = DigitRecognizerApp(model_path="model21.pth")
    app.run()
    print("Application Stopped!")