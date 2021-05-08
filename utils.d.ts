import * as tf from '@tensorflow/tfjs';
export declare const readInput: (img: any) => tf.Tensor<tf.Rank.R3>;
export declare const loadModel: () => Promise<(input: any) => tf.Tensor<tf.Rank.R1>>;
