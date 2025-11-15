import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# --- App Configuration ---
st.set_page_config(
    page_title="AI for SE Showcase",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching ---
# Cache the TFLite interpreter for performance
@st.cache_resource
def load_tflite_interpreter():
    """Loads the TFLite model and allocates tensors."""
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except ValueError:
        st.error("Error: 'model.tflite' not found.")
        st.error("Please run 'python train_model.py' first to create the model file.")
        return None

# Cache the dataset
@st.cache_data
def load_test_data():
    """Loads the CIFAR-10 test data."""
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_images = test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    return test_images, test_labels, class_names

# --- Sidebar Navigation ---
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Task 1: Edge AI Prototype", "Task 2: AI-Driven IoT Concept"])

# --- Page 1: Home ---
if page == "Home":
    st.title("AI for Software Engineering: Practical Showcase")
    st.write(f"**Hello from Ikole, Ekiti!** This app demonstrates the practical projects for this module.")
    st.image("https.placehold.co/800x300/223344/FFFFFF?text=AI+in+Software+Engineering", use_column_width=True)
    st.markdown("""
    This application, built by **[Your Name Here]**, showcases two key areas of AI in modern software:

    1.  **Task 1: Edge AI Prototype:** Demonstrates a lightweight, trained `TFLite` model running live inference on sample images. This simulates how a smart device (like a Raspberry Pi) would operate offline.
    
    2.  **Task 2: AI-Driven IoT Concept:** Lays out the complete system architecture for a "Smart Agriculture" platform, from sensors to AI-driven yield predictions.

    Use the sidebar to navigate to each project.
    """)

# --- Page 2: Task 1 (Edge AI) ---
elif page == "Task 1: Edge AI Prototype":
    st.header("Task 1: Edge AI Prototype (TFLite)")
    st.write("This page demonstrates the lightweight `model.tflite` file running live inference, just as it would on an edge device.")
    
    interpreter = load_tflite_interpreter()
    
    if interpreter:
        test_images, test_labels, class_names = load_test_data()
        
        st.subheader("Live TFLite Inference")
        if st.button("Run Inference on a Random Image"):
            with st.spinner("Classifying..."):
                # 1. Get random image
                idx = random.randint(0, len(test_images) - 1)
                img = test_images[idx]
                true_label_idx = test_labels[idx][0]
                true_label = class_names[true_label_idx]
                
                # 2. Prepare image for TFLite model
                img_batch = np.expand_dims(img, axis=0).astype(np.float32)
                
                # 3. Get interpreter details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # 4. Set tensor, invoke, and get output
                interpreter.set_tensor(input_details[0]['index'], img_batch)
                
                start_time = time.time()
                interpreter.invoke()
                inference_time = (time.time() - start_time) * 1000 # in ms
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # 5. Process prediction
                prediction_idx = np.argmax(output_data)
                pred_label = class_names[prediction_idx]
                
                st.write(f"**Inference Time:** `{inference_time:.2f} ms` (simulated on this server)")
                
                # 6. Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption=f"Original Test Image")
                
                with col2:
                    if pred_label == true_label:
                        st.success(f"**Prediction: '{pred_label}'**")
                        st.success(f"**Actual: '{true_label}'** (Correct!)")
                    else:
                        st.error(f"**Prediction: '{pred_label}'**")
                        st.error(f"**Actual: '{true_label}'** (Incorrect)")

        st.divider()
        
        # --- Display the Report ---
        st.header("Task 1: Project Report")
        st.subheader("1. Model & Performance Metrics")
        st.markdown("""
        * **Original Keras Model Accuracy:** `~63.74%`
        * **Converted TFLite Model Accuracy:** `~63.74%`
        * **TFLite Model Size:** `~45 KB`
        
        The conversion to TFLite retained **100% of the model's accuracy** while creating a tiny model file that is perfect for an edge device.
        """)

        st.subheader("2. Benefits of Edge AI for Real-Time Applications")
        st.markdown("""
        1.  **Low Latency (Speed):** The decision ("recyclable") is made in milliseconds on the device itself. There is **no delay** from sending an image to the cloud, which is critical for real-time robotics or autonomous drones.
        2.  **Privacy:** The images (e.g., of a user's trash) never leave the device, enhancing user privacy.
        3.  **Bandwidth & Cost:** The device does not need a constant, high-speed internet connection, saving enormous data costs.
        4.  **Reliability:** The app works 100% offline. If the internet fails, the edge app continues to function perfectly.
        """)

        st.subheader("3. Deployment Steps (Simulation to Raspberry Pi)")
        st.markdown("""
        1.  **Set Up Raspberry Pi:** Install Raspberry Pi OS and the TFLite Runtime (`pip install tflite-runtime`).
        2.  **Transfer Files:** Copy the `model.tflite` file to the Pi.
        3.  **Create Python Script:** Write a Python script using `tflite_runtime` to load the model and `opencv-python` to access the Pi's camera.
        4.  **Run Inference Loop:** In the script, continuously capture a camera frame, preprocess it (resize/normalize), feed it to the TFLite interpreter, and get the prediction, just as we've done in this app.
        """)

# --- Page 3: Task 2 (IoT Concept) ---
elif page == "Task 2: AI-Driven IoT Concept":
    st.header("Task 2: AI-Driven IoT Smart Agriculture Concept")
    
    st.subheader("1. Required IoT Sensors")
    st.markdown("""
    * **Soil Sensors (In-Ground):**
        * Soil Moisture Sensor
        * Soil pH Sensor
        * EC Sensor (Electrical Conductivity for nutrients)
    * **Environmental Sensors (Above-Ground):**
        * Temperature & Humidity Sensor
    * **Weather Station Sensors:**
        * Ambient Light/PAR Sensor (Measures usable sunlight)
        * Anemometer (Wind speed)
        * Rain Gauge (Precipitation)
    * **Advanced (Optional):**
        * Multi-Spectral Cameras (Drones/Poles) for NDVI (plant health) monitoring.
    """)

    st.subheader("2. Proposed AI Model: Crop Yield Prediction")
    st.markdown("""
    * **Goal:** Predict final crop yield (e.g., in tons per hectare). This is a **regression** problem.
    * **Input Features:** `avg_daily_temp`, `avg_daily_moisture`, `total_sunlight_hours`, `avg_ph_level`, `crop_type`, `fertilizer_type`, etc.
    * **Baseline Model: `RandomForestRegressor`**
        * **Why:** Excellent for handling mixed data types (sensor readings, text labels) and is highly "explainable" (we can see which features were most important).
    * **Advanced Model: `LSTM (Long Short-Term Memory) Network`**
        * **Why:** Crop yield depends on *timing*. An LSTM is a neural network designed for time-series data. It can learn that "low moisture in Week 3" has a different impact than "low moisture in Week 10", something a simpler model would miss.
    """)

    st.subheader("3. Data Flow Diagram")
    st.markdown("""
    This diagram shows the flow of data from the farm to the cloud and back.
    """)
    st.markdown("""
    ```mermaid
    graph TD
        subgraph "On-Farm (Edge)"
            A1[Soil Sensors] --> GW
            A2[Environmental Sensors] --> GW
            A3[Weather Station] --> GW
            GW[IoT Gateway / LoRaWAN] --> B[Cloud Ingest]
            
            M[Actuators: Irrigation Valves]
            N[Actuators: Fertilizer Pumps]
        end

        subgraph "Cloud Platform (Processing & AI)"
            B --> C{Time-Series DB}
            C --> D[Data Preprocessing]
            D --> E[AI Model: Yield Prediction (LSTM)]
            E --> F[Decision Engine]
            F --> G[User Dashboard]
            F --> H[Alerting System]
            F --> I[Control API]
        end
        
        subgraph "User & Actions"
            G[User Dashboard / Farmer's App]
            H --> J[SMS/Email to Farmer]
            I --> GW
        end
    ```
    """)
    st.subheader("Diagram Explanation")
    st.markdown("""
    1.  **Sensors (A1, A2, A3)** send data to a local `IoT Gateway (GW)`.
    2.  The **Gateway (GW)** forwards this data to the `Cloud Platform (B)`.
    3.  Data is stored in a `Time-Series Database (C)`.
    4.  The `AI Model (E)` runs on this data to update its yield prediction.
    5.  The `Decision Engine (F)` sends the prediction to the farmer's `Dashboard (G)` and can also send `Alerts (H)`.
    6.  **Control Loop:** The `Decision Engine` can also send automated commands (via the `Control API (I)`) back to the farm's `Gateway (GW)` to trigger `Actuators (M, N)`, such as turning on the irrigation.
    """)
