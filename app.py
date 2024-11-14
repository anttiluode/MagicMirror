import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageTk, ImageColor
from tkinter import (
    Tk,
    Label,
    Button,
    Entry,
    StringVar,
    Scale,
    HORIZONTAL,
    Radiobutton,
    OptionMenu,
    Frame,
    colorchooser,
    IntVar,
    Checkbutton,
    LEFT,
    RIGHT,
    TOP,
    BOTH,
    X,
    Scrollbar,
    Canvas,
)
from threading import Thread
import time
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIEdgeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("AI-Enhanced Edge Detection")
        self.master.geometry("1200x1000")  # Set a reasonable default size
        self.master.resizable(True, True)  # Allow window resizing

        # Initialize video capture
        self.cap = None
        self.camera_index = 0  # Default camera index
        self.available_cameras = self.get_camera_indices()

        # Default settings
        self.background_color = (0, 0, 0)  # Black
        self.threshold1 = 100
        self.threshold2 = 200  # Will be set as 2 * threshold1 in update_video
        self.dilation_iterations = 1
        self.use_edges_for_ai = IntVar(value=1)  # 1: Use Edge Image, 0: Use Original

        # AI Generation Parameters
        self.seed = StringVar(value="42")  # Default seed
        self.guidance_scale = StringVar(value="7.5")  # Default guidance scale

        # CV2 Manipulation Toggles
        self.enable_brightness = IntVar(value=0)
        self.enable_contrast = IntVar(value=0)
        self.enable_blur = IntVar(value=0)
        self.enable_grayscale = IntVar(value=0)
        self.enable_threshold = IntVar(value=0)
        self.enable_sharpen = IntVar(value=0)
        self.enable_sepia = IntVar(value=0)

        # Create Frames for better organization
        self.top_frame = Frame(master)
        self.top_frame.pack(side=TOP, fill=X, padx=10, pady=5)

        self.middle_frame = Frame(master)
        self.middle_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=5)

        self.bottom_frame = Frame(master)
        self.bottom_frame.pack(side=TOP, fill=X, padx=10, pady=5)

        # Status Label
        self.status_label = Label(master, text="Loading model, please wait...")
        self.status_label.pack(pady=5)

        # -------------------- Top Frame: Camera Controls --------------------
        # Dropdown for camera selection
        self.camera_var = StringVar(master)
        if self.available_cameras:
            self.camera_var.set(self.available_cameras[0])  # Default value
        else:
            self.camera_var.set("No Camera Found")
        self.camera_menu = OptionMenu(self.top_frame, self.camera_var, *self.available_cameras)
        self.camera_menu.grid(row=0, column=0, padx=5, pady=5)

        # Open Camera Button
        self.open_camera_button = Button(self.top_frame, text="Open Camera", command=self.open_camera)
        self.open_camera_button.grid(row=0, column=1, padx=5, pady=5)

        # -------------------- Middle Frame: Display Area --------------------
        # Subframes for video and AI images
        self.video_subframe = Frame(self.middle_frame)
        self.video_subframe.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        self.ai_subframe = Frame(self.middle_frame)
        self.ai_subframe.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

        # Label for live video with edges
        self.video_label = Label(self.video_subframe, text="Live Video Feed")
        self.video_label.pack(side=TOP, pady=5)

        self.video_display = Label(self.video_subframe)
        self.video_display.pack(side=TOP, fill=BOTH, expand=True)

        # Label for AI-generated image
        self.ai_image_label = Label(self.ai_subframe, text="AI-Generated Image")
        self.ai_image_label.pack(side=TOP, pady=5)

        self.ai_display = Label(self.ai_subframe)
        self.ai_display.pack(side=TOP, fill=BOTH, expand=True)

# -------------------- Bottom Frame: Controls --------------------
        # Prompt Entry
        self.prompt_var = StringVar()
        self.prompt_entry = Entry(self.bottom_frame, textvariable=self.prompt_var, width=80)
        self.prompt_entry.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        self.prompt_entry.insert(0, "A portrait of a person with ")

        # AI Input Selection - Radio Buttons for Original vs. Processed Image
        self.image_selection_var = IntVar(value=1)  # 1 for CV2 Processed Image, 0 for Original Camera Image

        # Radio button to choose the CV2 processed image for AI generation
        self.cv2_image_radio = Radiobutton(
            self.bottom_frame,
            text="Use Edge-Detected CV2 Processed Image for AI Style Transfer",
            variable=self.image_selection_var,
            value=1
        )
        self.cv2_image_radio.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Radio button to choose the original image for AI generation
        self.original_image_radio = Radiobutton(
            self.bottom_frame,
            text="Use Original Camera Image for AI Style Transfer",
            variable=self.image_selection_var,
            value=0
        )
        self.original_image_radio.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="w")

        # Start and Stop AI Rendering Buttons
        self.ai_buttons_frame = Frame(self.bottom_frame)
        self.ai_buttons_frame.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.start_ai_button = Button(self.ai_buttons_frame, text="Start AI Rendering", command=self.start_ai_rendering)
        self.start_ai_button.pack(side=LEFT, padx=5, pady=5)

        self.stop_ai_button = Button(self.ai_buttons_frame, text="Stop AI Rendering", command=self.stop_ai_rendering, state="disabled")
        self.stop_ai_button.pack(side=LEFT, padx=5, pady=5)

        # Style Transfer Strength Slider
        self.strength_var = StringVar(value="0.75")
        self.strength_scale = Scale(
            self.bottom_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=HORIZONTAL,
            label="Style Transfer Strength",
            variable=self.strength_var,
            length=300,
        )
        self.strength_scale.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="we")

        # Seed Entry
        self.seed_label = Label(self.bottom_frame, text="Seed:")
        self.seed_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.seed_entry = Entry(self.bottom_frame, textvariable=self.seed, width=20)
        self.seed_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Guidance Scale Slider (Temperature Equivalent)
        self.guidance_scale_label = Label(self.bottom_frame, text="Guidance Scale:")
        self.guidance_scale_label.grid(row=3, column=2, padx=5, pady=5, sticky="e")
        self.guidance_scale_slider = Scale(
            self.bottom_frame,
            from_=1.0,
            to=20.0,
            resolution=0.5,
            orient=HORIZONTAL,
            label="Guidance Scale",
            variable=self.guidance_scale,
            length=300,
        )
        self.guidance_scale_slider.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        # Create CV2 Options within a Canvas with Scrollbar
        self.cv2_canvas = Canvas(self.bottom_frame, height=300)
        self.cv2_canvas.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        self.cv2_scrollbar = Scrollbar(self.bottom_frame, orient="vertical", command=self.cv2_canvas.yview)
        self.cv2_scrollbar.grid(row=4, column=4, sticky="ns")

        self.cv2_canvas.configure(yscrollcommand=self.cv2_scrollbar.set)

        self.cv2_options_frame_inner = Frame(self.cv2_canvas)
        self.cv2_canvas.create_window((0, 0), window=self.cv2_options_frame_inner, anchor='nw')

        self.cv2_options_frame_inner.bind("<Configure>", lambda event: self.cv2_canvas.configure(scrollregion=self.cv2_canvas.bbox("all")))

        # CV2 Manipulation Options
        # Row 0: Brightness
        self.brightness_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Brightness Adjustment",
            variable=self.enable_brightness,
            command=self.toggle_brightness,
        )
        self.brightness_check.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.brightness_var = StringVar(value="0")
        self.brightness_scale = Scale(
            self.cv2_options_frame_inner,
            from_=-100,
            to=100,
            resolution=1,
            orient=HORIZONTAL,
            label="Brightness",
            variable=self.brightness_var,
            length=300,
        )
        self.brightness_scale.grid(row=0, column=1, padx=5, pady=2)
        self.brightness_scale.configure(state="disabled")

        # Row 1: Contrast
        self.contrast_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Contrast Adjustment",
            variable=self.enable_contrast,
            command=self.toggle_contrast,
        )
        self.contrast_check.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.contrast_var = StringVar(value="1.0")
        self.contrast_scale = Scale(
            self.cv2_options_frame_inner,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=HORIZONTAL,
            label="Contrast",
            variable=self.contrast_var,
            length=300,
        )
        self.contrast_scale.grid(row=1, column=1, padx=5, pady=2)
        self.contrast_scale.configure(state="disabled")

        # Row 2: Gaussian Blur
        self.blur_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Gaussian Blur",
            variable=self.enable_blur,
            command=self.toggle_blur,
        )
        self.blur_check.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        self.blur_var = StringVar(value="0")
        self.blur_scale = Scale(
            self.cv2_options_frame_inner,
            from_=0,
            to=10,
            resolution=1,
            orient=HORIZONTAL,
            label="Gaussian Blur Kernel Size",
            variable=self.blur_var,
            length=300,
        )
        self.blur_scale.grid(row=2, column=1, padx=5, pady=2)
        self.blur_scale.configure(state="disabled")

        # Row 3: Grayscale Conversion
        self.grayscale_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Grayscale Conversion",
            variable=self.enable_grayscale,
            command=self.toggle_grayscale,
        )
        self.grayscale_check.grid(row=3, column=0, sticky="w", padx=5, pady=2)

        # Row 4: Thresholding
        self.threshold_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Thresholding",
            variable=self.enable_threshold,
            command=self.toggle_threshold,
        )
        self.threshold_check.grid(row=4, column=0, sticky="w", padx=5, pady=2)

        self.threshold_value_var = StringVar(value="127")
        self.threshold_value_scale = Scale(
            self.cv2_options_frame_inner,
            from_=0,
            to=255,
            resolution=1,
            orient=HORIZONTAL,
            label="Threshold Value",
            variable=self.threshold_value_var,
            length=300,
        )
        self.threshold_value_scale.grid(row=4, column=1, padx=5, pady=2)
        self.threshold_value_scale.configure(state="disabled")

        # Row 5: Sharpening
        self.sharpen_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Sharpening",
            variable=self.enable_sharpen,
            command=self.toggle_sharpen,
        )
        self.sharpen_check.grid(row=5, column=0, sticky="w", padx=5, pady=2)

        # Row 6: Sepia Filter
        self.sepia_check = Checkbutton(
            self.cv2_options_frame_inner,
            text="Enable Sepia Filter",
            variable=self.enable_sepia,
            command=self.toggle_sepia,
        )
        self.sepia_check.grid(row=6, column=0, sticky="w", padx=5, pady=2)

        # Row 7: Dilation Iterations
        self.dilation_var = StringVar(value=str(self.dilation_iterations))
        self.dilation_scale = Scale(
            self.cv2_options_frame_inner,
            from_=0,
            to=10,
            resolution=1,
            orient=HORIZONTAL,
            label="Dilation Iterations",
            variable=self.dilation_var,
            length=300,
        )
        self.dilation_scale.grid(row=7, column=0, columnspan=2, padx=5, pady=2)

        # Expand columns in CV2 options frame
        self.cv2_options_frame_inner.columnconfigure(0, weight=1)
        self.cv2_options_frame_inner.columnconfigure(1, weight=1)

        # Take Picture and Close Buttons
        self.action_buttons_frame = Frame(self.bottom_frame)
        self.action_buttons_frame.grid(row=5, column=0, columnspan=5, padx=5, pady=5, sticky="w")

        # Take Picture Button
        self.take_picture_button = Button(self.action_buttons_frame, text="Take Picture", command=self.take_picture)
        self.take_picture_button.pack(side=LEFT, padx=5, pady=5)

        # Close Button
        self.close_button = Button(self.action_buttons_frame, text="Close", command=self.close)
        self.close_button.pack(side=LEFT, padx=5, pady=5)


        # -------------------- End of Layout Setup --------------------

        # Store the current frame
        self.current_frame = None
        self.live_generation = False

        # Load the Stable Diffusion model in a separate thread to avoid blocking the GUI
        self.pipe = None
        self.load_model_thread = Thread(target=self.load_model, daemon=True)
        self.load_model_thread.start()

        # Start the video feed
        self.update_video()

    def get_camera_indices(self):
        """Get the list of available camera indices."""
        indices = []
        for i in range(10):  # Check first 10 device indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    indices.append(f"Camera {i}")
                    logger.info(f"Camera {i} is available.")
                cap.release()
        if not indices:
            logger.warning("No cameras found.")
        return indices

    def open_camera(self):
        """Open the selected camera."""
        selected_camera = self.camera_var.get()
        if selected_camera.startswith("Camera"):
            try:
                self.camera_index = int(selected_camera.split(" ")[-1])  # Get the camera index
            except ValueError:
                logger.error("Invalid camera index selected.")
                self.status_label.config(text="Invalid camera index selected.")
                return

            if self.cap is not None:
                self.cap.release()  # Release any previously opened camera

            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Error: Could not open Camera {self.camera_index}.")
                self.status_label.config(text=f"Error: Could not open Camera {self.camera_index}.")
            else:
                logger.info(f"Camera {self.camera_index} opened successfully.")
                self.status_label.config(text=f"Camera {self.camera_index} opened successfully.")
        else:
            logger.error("No camera available to open.")
            self.status_label.config(text="No camera available to open.")

    def load_model(self):
        """Load the Stable Diffusion model."""
        logger.info("Loading Stable Diffusion model...")
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                logger.info("CUDA is available. Using GPU for inference.")
            else:
                device = "cpu"
                dtype = torch.float32
                logger.info("CUDA not available. Using CPU for inference.")

            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=dtype,
            )
            self.pipe = self.pipe.to(device)
            logger.info("Model loaded successfully!")
            self.status_label.config(text="Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.status_label.config(text="Error loading model.")
            sys.exit(1)

    def change_bg_color(self):
        """Open a color picker dialog to select background color."""
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code:
            # Convert hex to RGB tuple
            self.background_color = ImageColor.getrgb(color_code[1])
            logger.info(f"Background color changed to {self.background_color}.")

    # Toggle functions for CV2 options
    def toggle_brightness(self):
        if self.enable_brightness.get():
            self.brightness_scale.configure(state="normal")
        else:
            self.brightness_scale.configure(state="disabled")

    def toggle_contrast(self):
        if self.enable_contrast.get():
            self.contrast_scale.configure(state="normal")
        else:
            self.contrast_scale.configure(state="disabled")

    def toggle_blur(self):
        if self.enable_blur.get():
            self.blur_scale.configure(state="normal")
        else:
            self.blur_scale.configure(state="disabled")

    def toggle_grayscale(self):
        pass  # No additional controls needed

    def toggle_threshold(self):
        if self.enable_threshold.get():
            self.threshold_value_scale.configure(state="normal")
        else:
            self.threshold_value_scale.configure(state="disabled")

    def toggle_sharpen(self):
        pass  # No additional controls needed

    def toggle_sepia(self):
        pass  # No additional controls needed

    def apply_cv2_manipulations(self, frame):
        """Apply CV2 manipulations based on user toggles and slider values."""
        manipulated = frame.copy()

        # Brightness Adjustment
        if self.enable_brightness.get():
            try:
                brightness = int(self.brightness_var.get())
                # Convert to HSV to adjust brightness
                hsv = cv2.cvtColor(manipulated, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                if brightness >= 0:
                    lim = 255 - brightness
                    v[v > lim] = 255
                    v[v <= lim] = np.clip(v[v <= lim] + brightness, 0, 255)
                else:
                    lim = abs(brightness)
                    v[v < lim] = 0
                    v[v >= lim] = np.clip(v[v >= lim] + brightness, 0, 255)
                final_hsv = cv2.merge((h, s, v))
                manipulated = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                logger.info(f"Brightness adjusted by {brightness}.")
            except ValueError:
                logger.error("Invalid brightness value.")

        # Contrast Adjustment
        if self.enable_contrast.get():
            try:
                contrast = float(self.contrast_var.get())
                manipulated = cv2.convertScaleAbs(manipulated, alpha=contrast, beta=0)
                logger.info(f"Contrast adjusted by a factor of {contrast}.")
            except ValueError:
                logger.error("Invalid contrast value.")

        # Gaussian Blur
        if self.enable_blur.get():
            try:
                blur_size = int(self.blur_var.get())
                if blur_size % 2 == 0:
                    blur_size += 1  # Kernel size must be odd
                if blur_size > 1:
                    manipulated = cv2.GaussianBlur(manipulated, (blur_size, blur_size), 0)
                    logger.info(f"Applied Gaussian Blur with kernel size {blur_size}.")
            except ValueError:
                logger.error("Invalid blur kernel size.")

        # Grayscale Conversion
        if self.enable_grayscale.get():
            manipulated = cv2.cvtColor(manipulated, cv2.COLOR_BGR2GRAY)
            manipulated = cv2.cvtColor(manipulated, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
            logger.info("Applied Grayscale Conversion.")

        # Thresholding
        if self.enable_threshold.get():
            try:
                threshold_value = int(self.threshold_value_var.get())
                _, thresh = cv2.threshold(
                    cv2.cvtColor(manipulated, cv2.COLOR_BGR2GRAY),
                    threshold_value,
                    255,
                    cv2.THRESH_BINARY,
                )
                manipulated = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                logger.info(f"Applied Thresholding with value {threshold_value}.")
            except ValueError:
                logger.error("Invalid threshold value.")

        # Sharpening
        if self.enable_sharpen.get():
            sharpening_kernel = np.array(
                [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]
            )
            manipulated = cv2.filter2D(manipulated, -1, sharpening_kernel)
            logger.info("Applied Sharpening filter.")

        # Sepia Filter
        if self.enable_sepia.get():
            sepia_filter = np.array(
                [[0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]]
            )
            sepia_image = cv2.transform(manipulated, sepia_filter)
            sepia_image = np.clip(sepia_image, 0, 255)
            manipulated = sepia_image.astype(np.uint8)
            logger.info("Applied Sepia filter.")

        return manipulated



    def update_video(self):
        """Capture video frames, perform edge detection, and display them."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()  # Store the current frame for image generation

                # Get user settings
                try:
                    threshold1 = int(self.threshold_value_var.get())
                    self.threshold1 = threshold1
                    self.threshold2 = min(threshold1 * 2, 255)  # Ensure threshold2 does not exceed 255
                    self.dilation_iterations = int(self.dilation_var.get())
                except ValueError:
                    self.threshold1 = 100
                    self.threshold2 = 200
                    self.dilation_iterations = 1

                # Apply CV2 manipulations based on user settings
                processed_frame = self.apply_cv2_manipulations(frame)

                # Edge Detection using Canny
                gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_frame, threshold1=self.threshold1, threshold2=self.threshold2)

                # Apply dilation to control line thickness
                if self.dilation_iterations > 0:
                    kernel_size = 3  # You can make this adjustable if needed
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=self.dilation_iterations)

                # Convert edges to RGB and apply background color
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                background = np.full_like(edges_rgb, self.background_color, dtype=np.uint8)
                mask = edges > 0
                combined_image = np.where(mask[:, :, None], edges_rgb, background)

                # Convert the BGR frame to RGB for Tkinter
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Create a side-by-side image of original and processed
                combined_display = np.hstack((rgb_frame, combined_image))

                # Convert to PIL Image
                pil_image = Image.fromarray(combined_display)

                # Resize for display if necessary
                pil_image = pil_image.resize((800, 400))

                # Convert to ImageTk
                imgtk = ImageTk.PhotoImage(image=pil_image)

                # Update the video display
                self.video_display.imgtk = imgtk
                self.video_display.configure(image=imgtk)

        self.master.after(10, self.update_video)

    def start_ai_rendering(self):
        """Start live AI image generation."""
        if not self.live_generation:
            if self.pipe is None:
                logger.info("AI model is still loading. Please wait.")
                self.status_label.config(text="AI model is loading. Please wait...")
                return
            self.live_generation = True
            self.start_ai_button.config(text="Stop AI Rendering")
            self.stop_ai_button.config(state="normal")
            self.image_generation_thread = Thread(target=self.live_generate_ai_image, daemon=True)
            self.image_generation_thread.start()
            self.status_label.config(text="Live AI Image Generation Started.")
            logger.info("Live AI Image Generation Started.")
        else:
            self.stop_ai_rendering()

    def stop_ai_rendering(self):
        """Stop live AI image generation."""
        if self.live_generation:
            self.live_generation = False
            self.start_ai_button.config(text="Start AI Rendering")
            self.stop_ai_button.config(state="disabled")
            self.status_label.config(text="Live AI Image Generation Stopped.")
            logger.info("Live AI Image Generation Stopped.")

    def live_generate_ai_image(self):
        """Continuously generate AI images based on current edges or original image."""
        while self.live_generation:
            if self.current_frame is not None and self.pipe is not None:
                try:
                    # Choose image source based on user selection
                    if self.image_selection_var.get() == 1:  # Use Edge-Detected CV2 Processed Image
                        processed_frame = self.apply_cv2_manipulations(self.current_frame)  # Apply CV2 manipulations

                        # Apply edge detection
                        gray_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_image, self.threshold1, self.threshold2)

                        # Convert edges to PIL Image
                        edge_image = Image.fromarray(edges).convert("RGB")
                        ai_input = edge_image

                        logger.info("Using edge-detected image for AI generation.")

                    else:  # Use Original Camera Image or CV2-Manipulated Image without Edges
                        ai_input_frame = self.apply_cv2_manipulations(self.current_frame)
                        ai_input_frame_rgb = cv2.cvtColor(ai_input_frame, cv2.COLOR_BGR2RGB)
                        ai_input = Image.fromarray(ai_input_frame_rgb).resize((512, 512))

                        logger.info("Using original camera image or CV2 manipulated image for AI generation.")

                    # Get the strength and guidance_scale values from the sliders
                    try:
                        strength = float(self.strength_var.get())
                        strength = max(min(strength, 1.0), 0.1)  # Clamp between 0.1 and 1.0
                        guidance = float(self.guidance_scale.get())
                        guidance = max(min(guidance, 20.0), 1.0)  # Clamp between 1.0 and 20.0
                    except ValueError:
                        strength = 0.75  # Default value if conversion fails
                        guidance = 7.5  # Default guidance scale

                    # Set seed for reproducibility
                    try:
                        seed = int(self.seed.get())
                    except ValueError:
                        seed = 42  # Default seed if conversion fails
                        self.seed.set(str(seed))
                        logger.warning("Invalid seed value entered. Using default seed 42.")

                    # Generate the image using the AI model
                    generator = torch.Generator(device=self.pipe.device)
                    generator.manual_seed(seed)

                    result = self.pipe(
                        prompt=self.prompt_var.get(),
                        image=ai_input,
                        strength=strength,
                        guidance_scale=guidance,
                        generator=generator,
                    ).images[0]

                    # Update the AI image display
                    result_pil = result.resize((400, 400))
                    imgtk = ImageTk.PhotoImage(image=result_pil)
                    self.ai_display.imgtk = imgtk
                    self.ai_display.configure(image=imgtk)
                    logger.info("AI image generated and updated.")

                except Exception as e:
                    logger.error(f"Error during AI image generation: {e}")
                    self.status_label.config(text=f"Error: {e}")

                # Sleep for a short while to avoid overwhelming the system
                time.sleep(10)  # Adjust this time as necessary (e.g., 10 seconds)

    def take_picture(self):
        """Generate and display an AI image based on the current edges or original image."""
        if self.current_frame is not None and self.pipe is not None:
            try:
                # Choose image source based on user selection
                if self.image_selection_var.get() == 1:  # Use Edge-Detected CV2 Processed Image
                    processed_frame = self.apply_cv2_manipulations(self.current_frame)  # Apply CV2 manipulations

                    # Apply edge detection
                    gray_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_image, self.threshold1, self.threshold2)

                    # Convert edges to PIL Image
                    edge_image = Image.fromarray(edges).convert("RGB")
                    ai_input = edge_image

                    logger.info("Using edge-detected image for AI generation.")

                else:  # Use Original Camera Image or CV2-Manipulated Image without Edges
                    ai_input_frame = self.apply_cv2_manipulations(self.current_frame)
                    ai_input_frame_rgb = cv2.cvtColor(ai_input_frame, cv2.COLOR_BGR2RGB)
                    ai_input = Image.fromarray(ai_input_frame_rgb).resize((512, 512))

                    logger.info("Using original camera image or CV2 manipulated image for AI generation.")

                # Get the strength and guidance_scale values from the sliders
                try:
                    strength = float(self.strength_var.get())
                    strength = max(min(strength, 1.0), 0.1)  # Clamp between 0.1 and 1.0
                    guidance = float(self.guidance_scale.get())
                    guidance = max(min(guidance, 20.0), 1.0)  # Clamp between 1.0 and 20.0
                except ValueError:
                    strength = 0.75  # Default value if conversion fails
                    guidance = 7.5  # Default guidance scale

                # Set seed for reproducibility
                try:
                    seed = int(self.seed.get())
                except ValueError:
                    seed = 42  # Default seed if conversion fails
                    self.seed.set(str(seed))
                    logger.warning("Invalid seed value entered. Using default seed 42.")

                # Generate the image using the AI model
                generator = torch.Generator(device=self.pipe.device)
                generator.manual_seed(seed)

                result = self.pipe(
                    prompt=self.prompt_var.get(),
                    image=ai_input,
                    strength=strength,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]

                # Update the AI image display
                result_pil = result.resize((400, 400))
                imgtk = ImageTk.PhotoImage(image=result_pil)
                self.ai_display.imgtk = imgtk
                self.ai_display.configure(image=imgtk)
                logger.info("AI image generated and updated via Take Picture.")

                # Save the image with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"ai_generated_image_{timestamp}.png"
                result.save(filename)
                logger.info(f"AI-generated image saved as {filename}.")
                self.status_label.config(text=f"AI image saved as {filename}.")

            except Exception as e:
                logger.error(f"Error during AI image generation: {e}")
                self.status_label.config(text=f"Error: {e}")


    def close(self):
        """Release resources and close the application."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.master.quit()


if __name__ == "__main__":
    root = Tk()
    app = AIEdgeApp(root)
    root.mainloop()
