import { loadModel, readInput } from './utils';

import * as tf from '@tensorflow/tfjs';
require('@tensorflow/tfjs-node');

const Lefts = 'augment/left-aug';
const Rights = 'augment/right-aug';
const Ups = 'augment/up-aug';
const Downs = 'augment/down-aug';
const Others = 'augment/other-aug';

const Epochs = 300;
const BatchSize = 0.1;

const train = async () => {
  const mobileNet = await loadModel();
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [1024] }));
  model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));
  await model.compile({
    optimizer: tf.train.adam(1e-6),
    loss: tf.losses.sigmoidCrossEntropy,
    metrics: ['accuracy']
  });

  const lefts = require('fs')
    .readdirSync(Lefts)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Lefts}/${f}`);

  const rights = require('fs')
    .readdirSync(Rights)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Rights}/${f}`);

  const ups = require('fs')
    .readdirSync(Ups)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Ups}/${f}`);

  const downs = require('fs')
    .readdirSync(Downs)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Downs}/${f}`);

  const others = require('fs')
    .readdirSync(Others)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Others}/${f}`);

  console.log('Building the training set');
  const ys = tf.tensor2d(new Array(lefts.length).fill([1,0,0,0,0])
    .concat(new Array(rights.length).fill([0,1,0,0,0]))
    .concat(new Array(ups.length).fill([0,0,1,0,0]))
    .concat(new Array(downs.length).fill([0,0,0,1,0]))
    .concat(new Array(others.length).fill([0,0,0,0,1])),
    [lefts.length + rights.length + ups.length + downs.length + others.length, 5]);

  console.log('Getting the movements');
  const xs: tf.Tensor2D = tf.stack(
    lefts.map((path: string) => mobileNet(readInput(path)))
    .concat(rights.map((path: string) => mobileNet(readInput(path))))
    .concat(ups.map((path: string) => mobileNet(readInput(path))))
    .concat(downs.map((path: string) => mobileNet(readInput(path))))
    .concat(others.map((path: string) => mobileNet(readInput(path))))) as tf.Tensor2D;
  console.log('shape:', xs.shape);
  xs.print();

  console.log('Fitting the model');
  await model.fit(xs, ys, {
    epochs: Epochs,
    batchSize: parseInt(((lefts.length + rights.length + ups.length + downs.length + others.length) * BatchSize).toFixed(0)),
    callbacks: {
      onBatchEnd: async (_, logs) => {
        console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
        await tf.nextFrame();
      }
    }
  });

  await model.save('file://movements_simplified');
};

train();
