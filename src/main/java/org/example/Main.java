package org.example;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {
    public static void main(String[] args) {
        int batchSize = 10;
        int numClasses = 3;
        DataSetIterator iterator = new IrisDataSetIterator(batchSize,numClasses);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.1,0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(10).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(10)
                .nOut(numClasses)
                .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        int numEpochs = 100;
        for (int i = 0; i < numEpochs; i++){
            iterator.reset();
            model.fit(iterator);
        }

        iterator.reset();
        Evaluation evaluation = new Evaluation(numClasses);
        while(iterator.hasNext()){
            DataSet batch = iterator.next();
            model.rnnClearPreviousState();
            evaluation.eval(batch.getLabels(), model.output(batch.getFeatures()));
        }

        System.out.println(evaluation.stats());
    }
}