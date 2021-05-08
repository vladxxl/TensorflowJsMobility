"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var utils_1 = require("./utils");
var tf = require("@tensorflow/tfjs");
require('@tensorflow/tfjs-node');
var Lefts = 'augment/left-aug';
var Rights = 'augment/right-aug';
var Ups = 'augment/up-aug';
var Downs = 'augment/down-aug';
var Others = 'augment/other-aug';
var Epochs = 300;
var BatchSize = 0.1;
var train = function () { return __awaiter(_this, void 0, void 0, function () {
    var mobileNet, model, lefts, rights, ups, downs, others, ys, xs;
    var _this = this;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0: return [4 /*yield*/, utils_1.loadModel('./mobile-net/model.json')];
            case 1:
                mobileNet = _a.sent();
                model = tf.sequential();
                model.add(tf.layers.inputLayer({ inputShape: [1024] }));
                model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
                model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));
                return [4 /*yield*/, model.compile({
                        optimizer: tf.train.adam(1e-6),
                        loss: tf.losses.sigmoidCrossEntropy,
                        metrics: ['accuracy']
                    })];
            case 2:
                _a.sent();
                lefts = require('fs')
                    .readdirSync(Lefts)
                    .filter(function (f) { return f.endsWith('.jpg'); })
                    .map(function (f) { return Lefts + "/" + f; });
                rights = require('fs')
                    .readdirSync(Rights)
                    .filter(function (f) { return f.endsWith('.jpg'); })
                    .map(function (f) { return Rights + "/" + f; });
                ups = require('fs')
                    .readdirSync(Ups)
                    .filter(function (f) { return f.endsWith('.jpg'); })
                    .map(function (f) { return Ups + "/" + f; });
                downs = require('fs')
                    .readdirSync(Downs)
                    .filter(function (f) { return f.endsWith('.jpg'); })
                    .map(function (f) { return Downs + "/" + f; });
                others = require('fs')
                    .readdirSync(Others)
                    .filter(function (f) { return f.endsWith('.jpg'); })
                    .map(function (f) { return Others + "/" + f; });
                console.log('Building the training set');
                ys = tf.tensor2d(new Array(lefts.length).fill([1, 0, 0, 0, 0])
                    .concat(new Array(rights.length).fill([0, 1, 0, 0, 0]))
                    .concat(new Array(ups.length).fill([0, 0, 1, 0, 0]))
                    .concat(new Array(downs.length).fill([0, 0, 0, 1, 0]))
                    .concat(new Array(others.length).fill([0, 0, 0, 0, 1])), [lefts.length + rights.length + ups.length + downs.length + others.length, 5]);
                console.log('Getting the movements');
                xs = tf.stack(lefts.map(function (path) { return mobileNet(utils_1.readInput(path)); })
                    .concat(rights.map(function (path) { return mobileNet(utils_1.readInput(path)); }))
                    .concat(ups.map(function (path) { return mobileNet(utils_1.readInput(path)); }))
                    .concat(downs.map(function (path) { return mobileNet(utils_1.readInput(path)); }))
                    .concat(others.map(function (path) { return mobileNet(utils_1.readInput(path)); })));
                console.log('Fitting the model');
                return [4 /*yield*/, model.fit(xs, ys, {
                        epochs: Epochs,
                        batchSize: parseInt(((lefts.length + rights.length + ups.length + downs.length + others.length) * BatchSize).toFixed(0)),
                        callbacks: {
                            onBatchEnd: function (_, logs) { return __awaiter(_this, void 0, void 0, function () {
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0:
                                            console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
                                            return [4 /*yield*/, tf.nextFrame()];
                                        case 1:
                                            _a.sent();
                                            return [2 /*return*/];
                                    }
                                });
                            }); }
                        }
                    })];
            case 3:
                _a.sent();
                return [4 /*yield*/, model.save('file://model')];
            case 4:
                _a.sent();
                return [2 /*return*/];
        }
    });
}); };
train();
//# sourceMappingURL=train.js.map