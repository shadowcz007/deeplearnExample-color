//******* 1 准备环境 ********************************************************
//deeplearn的方法暴露到 window下，方便调用
Object.keys(deeplearn).forEach((k, i) => {
  window[k] = deeplearn[k];
});


//******* 2 创建训练数据 ********************************************************

    function generateRandomChannelValue() {
      return Math.ceil(Math.random() * 255) - 1
    }

    function computeComplementaryColor(rgbColor) {
      let r = rgbColor[0];
      let g = rgbColor[1];
      let b = rgbColor[2];

      r /= 255.0;
      g /= 255.0;
      b /= 255.0;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      let h = (max + min) / 2.0;
      let s = h;
      const l = h;

      if (max === min) {
        h = s = 0;
      } else {
        const d = max - min;
        s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));

        if (max === r && g >= b) {
          h = 1.0472 * (g - b) / d;
        } else if (max === r && g < b) {
          h = 1.0472 * (g - b) / d + 6.2832;
        } else if (max === g) {
          h = 1.0472 * (b - r) / d + 2.0944;
        } else if (max === b) {
          h = 1.0472 * (r - g) / d + 4.1888;
        }
      }

      h = h / 6.2832 * 360.0 + 0;

      h += 180;
      if (h > 360) {
        h -= 360;
      }
      h /= 360;

      if (s === 0) {
        r = g = b = l;
      } else {

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;

        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
      };

      function hue2rgb(p, q, t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };

      return [r, g, b].map(v => Math.round(v * 255));
    };


    //normalize inputs

    function normalizeColor(rgbColor) {

      return rgbColor.map(v => v / 255);

    };

    function denormalizeColor(rgbColor) {

      var colors = rgbColor.map(v => v * 255);

      return colors.map(v => Math.round(Math.max(Math.min(v, 255), 0)));
    };

//!!!!!!!! 输入 deeplearn到数据处理方式见 ComplementaryColorModel.generateTrainingData


//******* 3 构建图 ********************************************************
    function ComplementaryColorModel() {
      // 初始化，数学运算方式，可选CPU或GPU
      this.math = new NDArrayMathGPU();
      this.initialLearningRate= 0.042;// An optimizer with a certain initial learning rate. Used for training.
      this.optimizer= new SGDOptimizer(this.initialLearningRate); //创建一个梯度下降优化器对象

    };

    ComplementaryColorModel.prototype = {
      constructor: ComplementaryColorModel,

      setupSession: function() {

        /**
         *构建模型的图。 在训练之前调用这种方法 Constructs the graph of the model. Call this method before training.
         */

        const graph = new Graph();

        // This tensor contains the input. In this case, it is a scalar.
        this.inputTensor = graph.placeholder('input RGB value', [3]);

        // This tensor contains the target.
        this.targetTensor = graph.placeholder('output RGB value', [3]);

        // Create 3 fully connected layers, each with half the number of nodes of
        // the previous layer. The first one has 16 nodes.
        let fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64, true, true);

        // Create fully connected layer 1, which has 8 nodes.
        fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32, true, true);

        // Create fully connected layer 2, which has 4 nodes.
        fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16, true, true);


        this.predictionTensor = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3, true, true);

        // We will optimize using mean squared loss.
        this.costTensor = graph.meanSquaredCost(this.predictionTensor, this.targetTensor);

        // Create the session only after constructing the graph.
        this.session = new Session(graph, this.math);

      },
      generateTrainingData:function(exampleCount) {

        this.math.scope(() => {

          const rawInputs = new Array(exampleCount);

          for (let i = 0; i < exampleCount; i++) {
            rawInputs[i] = [
              generateRandomChannelValue(), generateRandomChannelValue(),
              generateRandomChannelValue()
            ];
          }

          // Store the data within Array1Ds so that learnjs can use it.
          const inputArray = rawInputs.map(c => Array1D.new(normalizeColor(c)));
          const targetArray = rawInputs.map(c => Array1D.new(normalizeColor(computeComplementaryColor(c))));

          // This provider will shuffle the training data (and will do so in a way
          // that does not separate the input-target relationship).
          const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
          const inputProvider = shuffledInputProviderBuilder.getInputProviders()[0];
          const targetProvider = shuffledInputProviderBuilder.getInputProviders()[1];

          // Maps tensors to InputProviders.
          this.feedEntries = [{
              tensor: this.inputTensor,
              data: inputProvider
            },
            {
              tensor: this.targetTensor,
              data: targetProvider
            }
          ];

        });
      },
      setupSessionAndTrainingData:function(exampleCount){

        this.setupSession();

        // Generate the data that will be used to train the model.
        this.generateTrainingData(exampleCount);

      },
      batchSize:300,// Each training batch will be on this many examples.
      train1Batch: function(shouldFetchCost) {
        /**
         * Trains one batch for one iteration. Call this method multiple times to
         * progressively train. Calling this function transfers data from the GPU in
         * order to obtain the current loss on training data.
         *
         * If shouldFetchCost is true, returns the mean cost across examples in the
         * batch. Otherwise, returns -1. We should only retrieve the cost now and then
         * because doing so requires transferring data from the GPU.
         */

        // Every 42 steps, lower the learning rate by 15%.
        const learningRate = this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));

        this.optimizer.setLearningRate(learningRate);

        // Train 1 batch.
        let costValue = -1;

        this.math.scope(() => {
          const cost = this.session.train(this.costTensor, this.feedEntries, this.batchSize, this.optimizer, shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

          if (!shouldFetchCost) {
            // We only train. We do not compute the cost.
            return;
          };

          // Compute the cost (by calling get), which requires transferring data
          // from the GPU.
          costValue = cost.get();
        });
        return costValue;
      },
      predict: function(rgbColor) {
        let complementColor = [];
        this.math.scope((keep, track) => {
          const mapping = [{
            tensor: this.inputTensor,
            data: Array1D.new(normalizeColor(rgbColor)),
          }];
          const evalOutput = this.session.eval(this.predictionTensor, mapping);
          const values = evalOutput.getValues();

          complementColor = denormalizeColor(Array.prototype.slice.call(values));

        });
        return complementColor;
      },
      createFullyConnectedLayer: function(graph, inputLayer, layerIndex, sizeOfThisLayer, includeRelu, includeBias) {

        return graph.layers.dense('fully_connected_' + layerIndex,
          inputLayer,
          sizeOfThisLayer,
          includeRelu ? (x) => graph.relu(x) : undefined,
          includeBias);
      }

    };
