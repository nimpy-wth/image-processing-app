# Image Processing Dashboard

This is an interactive web application built with Streamlit that allows users to upload images, use a webcam, or provide an image URL and apply various image processing filters and effects in real-time. The application provides a side-by-side comparison of the original and processed images, along with an interactive histogram of the resulting image.


## Features

-   **Multiple Image Sources**:
    -   Upload an image file directly from your computer (`jpg`, `jpeg`, `png`).
    -   Use your webcam to take a snapshot.
    -   Process a live feed directly from your webcam.
    -   Fetch an image from a direct URL.
-   **Sequential Image Processing**: Apply a series of processing operations. The order of selection matters and determines the processing pipeline.
-   **Configurable Operations**:
    -   **Grayscale**: Convert an image to shades of gray.
    -   **Blur**: Apply Gaussian blur with an adjustable kernel size.
    -   **Edge Detection**: Use the Canny algorithm with tunable upper and lower thresholds.
    -   **Thresholding**: Apply binary thresholding with an adjustable threshold value.
    -   **Sepia**: Add a classic, warm sepia tone to the image.
-   **Interactive Visualization**:
    -   **Side-by-Side View**: Instantly compare the original image with the processed one.
    -   **Dynamic Histogram**: View a Plotly-based histogram that updates in real-time to reflect the color/intensity distribution of the processed image.


## How to Set Up and Run the Project

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

You need to have **Python 3.7+** installed on your system.

### Installation

1.  **Clone the repository (or download the `app.py` file):**
    ```bash
    git clone https://github.com/nimpy-wth/image-processing-app.git
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

Once the dependencies are installed, you can run the Streamlit application with the following command in your terminal:

```bash
streamlit run app.py
```
Your web browser should automatically open a new tab with the running application.

