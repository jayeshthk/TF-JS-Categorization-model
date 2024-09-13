// Import TensorFlow.js for Node.js
const tf = require("@tensorflow/tfjs-node");

// Function to load the TensorFlow.js model
async function loadModel() {
  const model = await tf.loadGraphModel(
    ".https://s3bucket_url/tfjs_model/model.json"
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
