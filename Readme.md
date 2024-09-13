# TensorFlow.js Model for Node.js:

This project utilizes a TensorFlow.js model for making predictions in a Node.js/JS environment. Below is an overview of the process involved in training and converting the model.

## Overview

### Model Training

1. **Training the Model**:
   - The model is trained using TensorFlow or Keras in Python. This involves defining the model architecture (e.g., neural networks), compiling the model, and fitting it on the training data.
   - Training might involve tasks like classification, regression, or other types of predictive analytics, depending on your use case.

### Model Conversion

2. **Saving the Model**:

   - Once the model is trained, it is saved in a format compatible with TensorFlow.js. Typically, the model is saved as an HDF5 file (`model.h5`) or a TensorFlow SavedModel directory.

3. **Converting to TensorFlow.js Format**:

   - To use the model in a JavaScript environment, it must be converted to TensorFlow.js format. This is done using the TensorFlow.js Converter (`tensorflowjs_converter`).
   - The converter takes the saved model and outputs a `model.json` file along with binary weight files. These files can be hosted on a web server or cloud storage.

   ```bash
   tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model /path/to/saved_model /path/to/tfjs_model
   ```

   - For Keras models:

   ```bash
   tensorflowjs_converter --input_format=keras --output_format=tfjs_graph_model /path/to/model.h5 /path/to/tfjs_model
   ```

### Model Deployment

4. **Hosting the Model**:
   - The converted model files (`model.json` and associated weight files) are hosted on a server or cloud storage solution. Ensure that the files are accessible over HTTP/HTTPS.

### Using the Model in Node.js

5. **Loading and Predicting**:

   - In a Node.js environment, TensorFlow.js is used to load the model from the hosted location and perform predictions. The model is loaded via the TensorFlow.js Node.js API, and predictions are made based on input data processed into the appropriate format.

6. **Usage**:

- Load and Predict with TensorFlow.js Model
  Create a JavaScript file (e.g., predict.js) with the following code:

```js
const tf = require("@tensorflow/tfjs-node");

// Function to load the TensorFlow.js model
async function loadModel() {
  const model = await tf.loadGraphModel(
    "https://your-s3-bucket-url/model.json"
  );
  return model;
}

// Function to predict product category based on product name
async function predictProductCategory(productName) {
  const model = await loadModel();

  // Convert product name to character codes
  const input = productName.split("").map((char) => char.charCodeAt(0));

  // Create a 2D tensor (batch size 1)
  const inputTensor = tf.tensor([input], [1, input.length]);

  // Predict the category
  const prediction = model.predict(inputTensor);

  // Get the index of the highest predicted category
  const category = prediction.argMax(-1).dataSync()[0];

  return category;
}

// Example usage: Predict category for a given product name
predictProductCategory("Sony Camera").then((category) => {
  console.log("Predicted category:", category);
});
```

## Key Points

- **Model Training**: Done using TensorFlow or Keras in Python, involving defining and fitting the model.
- **Model Conversion**: Using TensorFlow.js Converter to convert the trained model to a format usable in JavaScript.
- **Model Hosting**: The converted model files are hosted online for access in web or Node.js environments.
- **Node.js Integration**: TensorFlow.js is used to load and interact with the model in a Node.js application.

## Additional Resources

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [TensorFlow.js Converter Guide](https://www.tensorflow.org/js/guide/conversion)

## files ref:

- ![tf_categorization.ipynb](tf_categorization.ipynb)- train model using colab (T4 GPU)
- ![product_categorization_model.h5](product_categorization_model.h5)- actual python model
- ![tensorflow.js](tensorflow.js) - loading convered model in JS
- ![pricerunner_aggregate.csv](pricerunner_aggregate.csv) - data file
- ![int_to_product_map_dict.json](int_to_product_map_dict.json) - output mapping file.

For further details, please refer to the TensorFlow.js documentation and guides related to model conversion and deployment.

## LICENSE

MIT-![LICENSE]()
