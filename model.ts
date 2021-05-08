import * as tf from '@tensorflow/tfjs';
import * as mobileNet from '@tensorflow-models/mobilenet';
import * as ui from './game/ui.js';

navigator.mediaDevices
  .getUserMedia({
    video: true,
    audio: false
  })
  .then(stream => {
    video.srcObject = stream;
  });

const video = document.getElementById('cam') as HTMLVideoElement;
const Layer = 'global_average_pooling2d_1';
const mobilenetInfer = m => (p): tf.Tensor<tf.Rank> => m.infer(p, Layer);
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const crop = document.getElementById('crop') as HTMLCanvasElement;

const ImageSize = {
  Width: 100,
  Height: 56
};

const grayscale = (canvas: HTMLCanvasElement) => {
  const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    data[i] = avg;
    data[i + 1] = avg;
    data[i + 2] = avg;
  }
  canvas.getContext('2d').putImageData(imageData, 0, 0);
};

let mobilenet: (p: any) => tf.Tensor<tf.Rank>;
tf.loadModel('http://localhost:5000/model.json').then(model => {
  mobileNet
      .load()
      .then((mn: any) => {
        mobilenet = mobilenetInfer(mn);
        document.getElementById('predict').addEventListener('click', () => ui.startPacman());
        document.getElementById('loading-page').style.display = 'none';
        console.log('MobileNet created');
      })
      .then(() => {
        setInterval(() => {
          canvas.getContext('2d').drawImage(video, 0, 0);
          crop.getContext('2d').drawImage(canvas, 0, 0, ImageSize.Width, ImageSize.Height);

          crop
              .getContext('2d')
              .drawImage(
                  canvas,
                  0,
                  0,
                  canvas.width,
                  canvas.width / (ImageSize.Width / ImageSize.Height),
                  0,
                  0,
                  ImageSize.Width,
                  ImageSize.Height
              );

          grayscale(crop);

          // Predicted values
          const [left, right, up, down, nothing] = Array.from((model.predict(
              mobilenet(tf.fromPixels(crop))
          ) as tf.Tensor1D).dataSync() as Float32Array);

          if (nothing >= 0.5) {
            document.getElementById('move').textContent = '- - - - -';
            return;
          }
          console.log(left.toFixed(2), right.toFixed(2), up.toFixed(2), down.toFixed(2), nothing.toFixed(2));

          if (up > left && up > right && up > down && up >= 0.8) {
            console.log('%cUp: ' + up.toFixed(2), 'color: blue; font-size: 30px');
            document.getElementById('move').textContent = 'Up';
            ui.predictClass(0);
            return;
          }

          if (down > left && down > right && down > up && down >= 0.8) {
            console.log('%cDown: ' + down.toFixed(2), 'color: blue; font-size: 30px');
            document.getElementById('move').textContent = 'Down';
            ui.predictClass(1);
            return;
          }

          if (left > right && left > up && left > down && left >= 0.5) {
            console.log('%cLeft: ' + left.toFixed(2), 'color: red; font-size: 30px');
            document.getElementById('move').textContent = 'Left';
            ui.predictClass(2);
            return;
          }

          if (right > left && right > up && right > down && right >= 0.5) {
            console.log('%cRight: ' + right.toFixed(2), 'color: red; font-size: 30px');
            document.getElementById('move').textContent = 'Right';
            ui.predictClass(3);
            return;
          }

        }, 250);
      });
});


