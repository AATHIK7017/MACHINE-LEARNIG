from ipywidgets import widgets, VBox, HBox, Label, Button, Output
from IPython.display import display
import pandas as pd
import numpy as np

# Assuming label_encoders and model are already defined from previous cells

# Create ipywidgets for each input feature
company_widget = widgets.Dropdown(options=label_encoders['Company'].classes_.tolist(), description='Company:')
type_widget = widgets.Dropdown(options=label_encoders['Type'].classes_.tolist(), description='Type:')
inches_widget = widgets.FloatText(description='Screen Size (inches):')
width_widget = widgets.IntText(description='Resolution Width:')
height_widget = widgets.IntText(description='Resolution Height:')
cpu_widget = widgets.Dropdown(options=label_encoders['Cpu'].classes_.tolist(), description='CPU:')
ram_widget = widgets.IntText(description='RAM (GB):')
memory_widget = widgets.IntText(description='Memory (GB):')
gpu_widget = widgets.Dropdown(options=label_encoders['Gpu'].classes_.tolist(), description='GPU:')
os_widget = widgets.Dropdown(options=label_encoders['Operating System'].classes_.tolist(), description='Operating System:')
weight_widget = widgets.FloatText(description='Weight (kg):')

# Create a button to trigger prediction
predict_button = widgets.Button(description='Predict Price')
output_widget = Output()

# Function to call when the button is clicked
def on_predict_button_clicked(b):
    with output_widget:
        output_widget.clear_output()
        try:
            # Get values from widgets
            company = company_widget.value
            type_name = type_widget.value
            inches = inches_widget.value
            width = width_widget.value
            height = height_widget.value
            cpu = cpu_widget.value
            ram = ram_widget.value
            memory = memory_widget.value
            gpu = gpu_widget.value
            os_name = os_widget.value
            weight = weight_widget.value

            # Encode categorical features using the fitted label encoders
            company_encoded = label_encoders['Company'].transform([company])[0]
            type_encoded = label_encoders['Type'].transform([type_name])[0]
            cpu_encoded = label_encoders['Cpu'].transform([cpu])[0]
            gpu_encoded = label_encoders['Gpu'].transform([gpu])[0]
            os_encoded = label_encoders['Operating System'].transform([os_name])[0]

            # Calculate resolution
            resolution = width * height

            # Create a DataFrame for prediction
            # Ensure the order of columns matches the training data
            input_data = pd.DataFrame([[
                company_encoded,
                type_encoded,
                inches,
                cpu_encoded,
                ram,
                memory,
                gpu_encoded,
                os_encoded,
                weight,
                resolution
            ]], columns=X_train.columns) # Use X_train.columns to maintain order and names

            # Predict the price
            predicted_price = model.predict(input_data)[0]

            print(f"Predicted Price: €{predicted_price:.2f}")

        except ValueError as e:
            print(f"Error: {e}. Please ensure all inputs are valid.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# Link the button click to the prediction function
predict_button.on_click(on_predict_button_clicked)

# Arrange widgets in a VBox
input_widgets = VBox([
    company_widget,
    type_widget,
    inches_widget,
    width_widget,
    height_widget,
    cpu_widget,
    ram_widget,
    memory_widget,
    gpu_widget,
    os_widget,
    weight_widget,
    predict_button
])

# Display the widgets and output area
display(input_widgets, output_widget)
