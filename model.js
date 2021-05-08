"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs");
var mobileNet = require("@tensorflow-models/mobilenet");
var ui = require("./game/ui.js");
navigator.mediaDevices
    .getUserMedia({
    video: true,
    audio: false
})
    .then(function (stream) {
    video.srcObject = stream;
});
var video = document.getElementById('cam');
var Layer = 'global_average_pooling2d_1';
var mobilenetInfer = function (m) { return function (p) { return m.infer(p, Layer); }; };
var canvas = document.getElementById('canvas');
var crop = document.getElementById('crop');
var ImageSize = {
    Width: 100,
    Height: 56
};
var grayscale = function (canvas) {
    var imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    var data = imageData.data;
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;
        data[i + 1] = avg;
        data[i + 2] = avg;
    }
    canvas.getContext('2d').putImageData(imageData, 0, 0);
};
var mobilenet;
tf.loadModel('http://localhost:5000/model.json').then(function (model) {
    mobileNet
        .load()
        .then(function (mn) {
        mobilenet = mobilenetInfer(mn);
        document.getElementById('predict').addEventListener('click', function () { return ui.startPacman(); });
        document.getElementById('loading-page').style.display = 'none';
        console.log('MobileNet created');
    })
        .then(function () {
        setInterval(function () {
            canvas.getContext('2d').drawImage(video, 0, 0);
            crop.getContext('2d').drawImage(canvas, 0, 0, ImageSize.Width, ImageSize.Height);
            crop
                .getContext('2d')
                .drawImage(canvas, 0, 0, canvas.width, canvas.width / (ImageSize.Width / ImageSize.Height), 0, 0, ImageSize.Width, ImageSize.Height);
            grayscale(crop);
            // Predicted values
            var _a = Array.from(model.predict(mobilenet(tf.fromPixels(crop))).dataSync()), left = _a[0], right = _a[1], up = _a[2], down = _a[3], nothing = _a[4];
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
//# sourceMappingURL=model.js.map