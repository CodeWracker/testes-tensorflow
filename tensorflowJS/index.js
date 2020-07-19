const tf = require("@tensorflow/tfjs");

let tensorTrainX;
let tensorTrainY;
let tensorTestX;
let tensorTestY;
trainModel();
async function buildDataSet() {
  const trainDataSet = [
    [10, 32],
    [15, 33],
    [0, 2],
    [9, 66],
  ];
  const trainX = tf.tensor2d(trainDataSet).toFloat().div(200.0);

  const trainLabels = [0, 1, 6, 9];
  const trainY = tf.tensor1d(trainLabels);

  const testDataSet = [
    [13, 6],
    [55, 150],
    [99, 33],
  ];
  const testX = tf.tensor2d(testDataSet).toFloat().div(200.0);

  const testLabels = [1, 9, 0];
  const testY = tf.tensor1d(testLabels);

  console.log(trainX.shape);
  tensorTrainX = trainX;
  tensorTrainY = trainY;
  tensorTestX = testX;
  tensorTestY = testY;
}
async function trainModel() {
  await buildDataSet().then(async () => {
    tensorTrainX.print();
    tensorTrainY.print();
    tensorTestX.print();
    tensorTestY.print();
    const model = tf.sequential();
    const learningRate = 0.01;
    const optimizer = tf.train.adam(learningRate);

    model.add(
      tf.layers.dense({
        units: 2,
        activation: "sigmoid",
        inputShape: [tensorTrainX.shape[1]],
      })
    );
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({
      optimizer: optimizer,
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    });
    const history = await model.fit(tensorTrainX, tensorTrainY, {
      epochs: 5,
      validationData: [tensorTestX, tensorTestY],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log("Epoch: " + epoch + " loss: " + logs.loss);
          await tf.nextFrame();
        },
      },
    });
  });
}
