import os
os.system('pip install Pillow')
from edge import *
from filtering import *
from operations import *
from PIL import Image, ImageTk
from segmentation import *
from simple_operations import *
from tkinter import filedialog, simpledialog
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk


def plt_to_pil(data, title):
    plt.figure(figsize=(10, 6))
    plt.bar(range(256), data, color='blue', width=1)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG')
    plt.close()

    buffer.seek(0)
    image = Image.open(buffer)
    image.thumbnail((800, 800))
    return image


def pil_to_cv2(pil_image):
    return np.array(pil_image)


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2_image)


class ImageProcessingApp:

    buttons = {
        "Threshold": 2,
        "Histogram": 3,
        "Edge": 4,
        "Segmentation": 5,
        "Halftone": 6,
        "Filtering": 7,
        "Operations": 8,
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Processing Application")
        self.frame = tk.Frame(self.root)
        self.images = []
        self.new_image_window = None
        self.command = None
        self.image_index = 0
        self.num_images = 0
        self.image_name = None
        self.output_name = None
        self.input_image = None
        self.reset_frame()
        self.window_one()

    def reset_frame(self):
        self.frame.destroy()
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH)
        self.create_top_frame()
        self.main_frame = tk.Frame(self.frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def create_top_frame(self):
        self.top_frame = tk.Frame(self.frame)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.next_button = tk.Button(
            self.top_frame, text="Next", state="disabled")
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.previous_button = tk.Button(
            self.top_frame, text="Previous", state="disabled")
        self.previous_button.pack(side=tk.LEFT, padx=5, pady=5)

    def window_one(self):
        self.reset_frame()
        tk.Button(self.main_frame, text="Upload Image",
                  command=self.upload_image).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Show Image",
                  command=self.image_window, state="normal" if self.input_image else "disabled").pack(fill=tk.X, padx=10, pady=10)
        if self.input_image:
            self.next_button.config(state="normal", command=self.window_two)

    def window_two(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_one)
        for key, value in ImageProcessingApp.buttons.items():
            tk.Button(self.main_frame, text=key, command=lambda value=value: self.action(
                value)).pack(fill=tk.X, padx=10, pady=10)
        if self.command:
            self.next_button.config(state="normal", command=self.command)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image_name = tk.StringVar(
                value=os.path.basename(os.path.splitext(file_path)[0]))
            self.output_name = tk.StringVar(value=self.image_name.get())
            self.input_image = gray_scale(Image.open(file_path))
            self.input_image.thumbnail((800, 800))
            self.image_window()
            self.window_two()

    def on_image_window_close(self):
        self.new_image_window.destroy()
        self.new_image_window = None

    def image_window(self):
        if self.new_image_window is None:
            self.new_image_window = tk.Toplevel(self.root)
            self.new_image_window.title("Images")

            self.new_image_window_left = tk.Frame(self.new_image_window)
            self.new_image_window_right = tk.Frame(self.new_image_window)

            self.new_image_window_left.pack(
                side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
            self.new_image_window_right.pack(
                side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

            self.left_top_label = tk.Label(
                self.new_image_window_left, textvariable=self.image_name)
            self.right_top_label = tk.Label(
                self.new_image_window_right, textvariable=self.output_name)

            self.left_top_label.pack(padx=10, pady=10, fill=tk.X)
            self.right_top_label.pack(padx=10, pady=10, fill=tk.X)

            self.input_image_tk = ImageTk.PhotoImage(self.input_image)

            self.left_mid_label = tk.Label(self.new_image_window_left)
            self.right_mid_label = tk.Label(self.new_image_window_right)

            self.left_mid_label.config(image=self.input_image_tk)
            self.left_mid_label.image = self.input_image_tk

            if self.num_images == 0:
                self.right_mid_label.config(image=self.input_image_tk)
                self.right_mid_label.image = self.input_image_tk
            else:
                self.output_image_tk = ImageTk.PhotoImage(
                    self.images[self.image_index])

                self.right_mid_label.config(image=self.output_image_tk)
                self.right_mid_label.image = self.output_image_tk

            self.left_mid_label.pack(padx=10, pady=10)
            self.right_mid_label.pack(padx=10, pady=10)

            self.left_single_button = tk.Button(
                self.new_image_window_left, text="Use output", command=self.use_output)
            self.left_single_button.pack(padx=10, pady=10)

            self.right_previous_button = tk.Button(
                self.new_image_window_right, text="Previous", state="disabled", command=lambda: self.change_index(False))
            self.right_previous_button.pack(side=tk.LEFT, padx=30, pady=10)
            self.right_next_button = tk.Button(
                self.new_image_window_right, text="next", state="disabled", command=self.change_index)
            self.right_next_button.pack(side=tk.RIGHT, padx=30, pady=10)

            if self.image_index < self.num_images - 1:
                self.right_next_button.config(state="normal")
            if self.image_index > 0:
                self.right_previous_button.config(state="normal")

            self.new_image_window.protocol(
                "WM_DELETE_WINDOW", self.on_image_window_close)

    def action(self, value):
        windows = [self.edge_window, self.segmentation_window,
                   self.halftone_window, self.filter_window, self.operations_window]
        if value < 4:
            image = pil_to_cv2(self.input_image)
            if value == 2:
                self.command = None
                self.window_two()
                threshold, image = calculate_threshold(image)
                threshold = int(threshold)
                self.output_name.set(self.image_name.get(
                ) + f"@calculate_threshold({threshold})")
                self.images = [image]
                self.image_index = 0
                self.num_images = 1
            else:
                self.command = None
                self.window_two()
                image, graph1, graph2 = Histogram_equalization(image)
                self.output_name.set(
                    self.image_name.get() + f"@histogram_equalization")
                self.images = [
                    cv2_to_pil(image),
                    plt_to_pil(graph1, "Histogram level"),
                    plt_to_pil(graph2, "Histogram equalization")
                ]
                self.image_index = 0
                self.num_images = 3
                self.right_next_button.config(state="normal")
            self.output_image_tk = ImageTk.PhotoImage(
                self.images[self.image_index])
            self.right_mid_label.config(image=self.output_image_tk)
            self.right_mid_label.image = self.output_image_tk
        else:
            self.command = windows[value - 4]
            self.next_button.config(state="normal", command=self.command)
            self.command()

    def image_operations(self, operation, name):
        if operation is invert:
            self.right_next_button.config(state="disabled")
            self.right_previous_button.config(state="disabled")
            image = pil_to_cv2(self.input_image)
            self.num_images = 1
            self.image_index = 0
            image = operation(image)
            self.output_name.set(self.image_name.get() + f"@{name}")
            self.images = [cv2_to_pil(image)]
        else:
            self.right_next_button.config(state="disabled")
            self.right_previous_button.config(state="disabled")
            image = pil_to_cv2(self.input_image)
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if file_path:
                self.num_images = 1
                self.image_index = 0
                new_image_name = os.path.basename(
                    os.path.splitext(file_path)[0])
                self.output_name.set(
                    self.image_name.get() + f"@{name}({new_image_name})")
                input_image = gray_scale(Image.open(file_path))
                input_image.thumbnail((800, 800))
                input_image = pil_to_cv2(input_image)
                self.images = [cv2_to_pil(operation(image, input_image))]
        self.output_image_tk = ImageTk.PhotoImage(
            self.images[self.image_index])
        self.right_mid_label.config(image=self.output_image_tk)
        self.right_mid_label.image = self.output_image_tk

    def operate_on_image(self, method, name, variables=[]):
        self.right_next_button.config(state="disabled")
        self.right_previous_button.config(state="disabled")
        image = pil_to_cv2(self.input_image)
        values = []
        for variable in variables:
            current_value = simpledialog.askinteger(
                variable, "Enter value in range [0, 255]", maxvalue=255, minvalue=0)
            if current_value is None:
                return
            values.append(current_value)
        self.output_name.set(self.image_name.get() + f"@{name}")
        if method is kirsch_edge_detection:
            self.num_images = 1
            self.image_index = 0
            image, direction = method(image)
            self.images = [cv2_to_pil(image)]
            self.output_name.set(self.output_name.get() +
                                 f"--dir({direction})")
        elif method is difference_of_gaussians_edge_detection:
            difference_of_gaussians, blured_image1, blured_image2 = method(
                image)
            self.images = [cv2_to_pil(difference_of_gaussians), cv2_to_pil(
                blured_image1), cv2_to_pil(blured_image2)]
            self.num_images = 3
            self.image_index = 0
            self.right_next_button.config(state="normal")
        else:
            self.num_images = 1
            self.image_index = 0
            if len(values) > 0:
                self.images = [cv2_to_pil(method(image, *values))]
            else:
                self.images = [cv2_to_pil(method(image))]
        self.output_image_tk = ImageTk.PhotoImage(
            self.images[self.image_index])
        self.right_mid_label.config(image=self.output_image_tk)
        self.right_mid_label.image = self.output_image_tk

    def edge_window(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_two)
        tk.Button(self.main_frame, text="Sobel", command=lambda: self.operate_on_image(
            sobel_edge_detection, "Sobel", ["threshold"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Prewitt", command=lambda: self.operate_on_image(
            prewitt_edge_detection, "Prewitt", ["threshold"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Range", command=lambda: self.operate_on_image(
            range_edge_detection, "Range")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Contrast", command=lambda: self.operate_on_image(
            contrast_edge_detection, "Contrast")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Variance", command=lambda: self.operate_on_image(
            variance_edge_detection, "Variance")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Difference", command=lambda: self.operate_on_image(
            difference_edge_detection, "Difference", ["threshold"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Homogeneity", command=lambda: self.operate_on_image(
            homogeneity_edge_detection, "Homogeneity", ["threshold"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Kirsch", command=lambda: self.operate_on_image(
            kirsch_edge_detection, "Kirsch")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Difference of gaussians", command=lambda: self.operate_on_image(
            difference_of_gaussians_edge_detection, "Difference of gaussians")).pack(fill=tk.X, padx=10, pady=10)

    def segmentation_window(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_two)
        tk.Button(self.main_frame, text="Manual", command=lambda: self.operate_on_image(
            manual_segmentation, "Manual", ["low_t", "high_t", "value"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Histogram peak", command=lambda: self.operate_on_image(
            histogram_peak_segmentation, "Histogram peak")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Histogram valley", command=lambda: self.operate_on_image(
            histogram_valley_segmentation, "Histogram valley")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Adaptive", command=lambda: self.operate_on_image(
            adaptive_histogram_segmentation, "Adaptive")).pack(fill=tk.X, padx=10, pady=10)

    def halftone_window(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_two)
        tk.Button(self.main_frame, text="Simple", command=lambda: self.operate_on_image(
            Simple_halftone, "Simple", ["threshold"])).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="halftone", command=lambda: self.operate_on_image(
            Halftone, "Halftone")).pack(fill=tk.X, padx=10, pady=10)

    def filter_window(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_two)
        tk.Button(self.main_frame, text="low pass", command=lambda: self.operate_on_image(
            lowpass, "low pass")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="high pass", command=lambda: self.operate_on_image(
            highpass, "high pass")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="median pass", command=lambda: self.operate_on_image(
            median_filter, "median pass")).pack(fill=tk.X, padx=10, pady=10)

    def operations_window(self):
        self.reset_frame()
        self.previous_button.config(state="normal", command=self.window_two)
        tk.Button(self.main_frame, text="Add image", command=lambda: self.image_operations(
            add, "Add image")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Subtract image", command=lambda: self.image_operations(
            sub, "Subtract image")).pack(fill=tk.X, padx=10, pady=10)
        tk.Button(self.main_frame, text="Invert image", command=lambda: self.image_operations(
            invert, "Invert image")).pack(fill=tk.X, padx=10, pady=10)

    def change_index(self, forward=True):
        if forward:
            self.image_index = self.image_index + 1
        else:
            self.image_index = self.image_index - 1
        self.right_next_button.config(
            state="normal" if self.image_index + 1 < self.num_images else "disabled")
        self.right_previous_button.config(
            state="normal" if self.image_index > 0 else "disabled")
        self.output_image_tk = ImageTk.PhotoImage(
            self.images[self.image_index])
        self.right_mid_label.config(image=self.output_image_tk)
        self.right_mid_label.image = self.output_image_tk

    def use_output(self):
        if self.image_index < self.num_images:
            self.input_image = self.images[self.image_index]
            self.input_image_tk = ImageTk.PhotoImage(self.input_image)
            self.image_name.set(self.output_name.get())
            self.left_mid_label.config(image=self.input_image_tk)
            self.left_mid_label.image = self.input_image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
