package com.explore.java_dl4j_image_classification;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;

public class ImageClassification {

	// Data train path -> data will be used to feed the model during training
	private static final File trainPath = new File("src\\main\\resources\\dataset\\train\\");

	// Data test path -> data will be used to evaluate the model for each iteration during training
	private static final File testPath = new File("src\\main\\resources\\dataset\\test\\");

	// Data evaluation path -> data will be used to evaluate the model after the model has been built
	private static final File evaluationPath = new File("src\\main\\resources\\dataset\\evaluation\\");

	// Model path
	private static File modelPath = new File("src\\main\\resources\\model\\image_classification_model.zip");

	public static void main(String[] args) throws Exception {
		int height = 28;
		int width = 28;
		int channel = 1; //Grayscale
		int rngseed = 12345;
		Random randNumGen = new Random(rngseed);
		int batchSize = 50; // Number of data to be trained for each iteration
		int output = 2;
		int epoch = 350;

		FileSplit trainData = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit testData = new FileSplit(testPath, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit evaluationData = new FileSplit(evaluationPath, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

		// Extract the parent path as the image label
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		// Set params for train data
		ImageRecordReader trainRR = new ImageRecordReader(height, width, channel, labelMaker);
		trainRR.initialize(trainData);
		DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, output);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(trainIter);
		trainIter.setPreProcessor(scaler);

		if (modelPath.exists()) {
			System.out.println("Model found!\nLoad model.....");
			// Set params for evaluation data
			MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
			model.getLabels();
			trainRR.initialize(evaluationData);
			DataSetIterator evalIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, output);
			scaler.fit(evalIter);
			evalIter.setPreProcessor(scaler);

			List<String> labelList = Arrays.asList("bee", "spider");

			// Example on how to get predict results with trained model
			evalIter.reset();
			DataSet evalDataSet = evalIter.next();
			evalDataSet.setLabelNames(labelList);
			String expectedResult = evalDataSet.getLabelName(0);
			List<String> predict = model.predict(evalDataSet);
			String modelResult = predict.get(0);

			System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted as "
					+ modelResult + "\n\n");

		} else {
			System.out.println("Model is not found!\nCreate model.....");
			// Set params for test data
			ImageRecordReader testRR = new ImageRecordReader(height, width, channel, labelMaker);
			testRR.initialize(testData);
			DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, output);
			testIter.setPreProcessor(scaler);

			// kernel, stride, pad
			ConvolutionLayer layer0 = new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 1, 1 },
					new int[] { 0, 0 }).nIn(channel)
					.nOut(50)
					.stride(1, 1)
					.padding(2, 2)
					.name("First convolution layer").activation(Activation.RELU).biasInit(0)
					.build();

			LocalResponseNormalization layer1 = new LocalResponseNormalization.Builder().name("lrn1").build();

			SubsamplingLayer layer2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					.kernelSize(3, 3)
					.stride(2, 2)
					.name("First subsampling layer")
					.build();

			ConvolutionLayer layer3 = new ConvolutionLayer.Builder(5, 5)
					.nIn(50)
					.nOut(70)
					.stride(1, 1)
					.padding(2, 2)
					.name("Second convolution layer")
					.activation(Activation.RELU)
					.biasInit(1)
					.build();

			LocalResponseNormalization layer4 = new LocalResponseNormalization.Builder()
					.name("lrn2")
					.build();

			SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					.kernelSize(3, 3)
					.stride(2, 2)
					.name("Second subsampling layer")
					.build();

			ConvolutionLayer layer6 = new ConvolutionLayer.Builder(3, 3)
					.nIn(70)
					.nOut(90)
					.stride(1, 1)
					.padding(2, 2)
					.name("Third convolution layer")
					.activation(Activation.RELU)
					.biasInit(1)
					.build();

			LocalResponseNormalization layer7 = new LocalResponseNormalization.Builder()
					.name("lrn3")
					.build();

			SubsamplingLayer layer8 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3)
					.stride(2, 2).name("Third subsampling layer").build();

			DenseLayer layer9 = new DenseLayer.Builder().activation(Activation.RELU).nOut(140).name("First DenseLayer")
					.biasInit(1).dropOut(0.5).dist(new GaussianDistribution(0, 0.005)).build();

			OutputLayer layer10 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.activation(Activation.SOFTMAX)
					.name("Output").nIn(140).nOut(output)
					.build();

			// Fully Connected Layer Config
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
					.dist(new NormalDistribution(0.0, 0.01)).seed(rngseed)
					.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.updater(new Nesterovs(0.006, 0.9))
					.l2(1e-4)
					.miniBatch(false)
					.list()
					.layer(0, layer0)
					.layer(1, layer1)
					.layer(2, layer2)
					.layer(3, layer3)
					.layer(4, layer4)
					.layer(5, layer5)
					.layer(6, layer6)
					.layer(7, layer7)
					.layer(8, layer8)
					.layer(9, layer9)
					.layer(10, layer10)
					.setInputType(InputType.convolutional(28, 28, 1)) // 28 x 28 images having 1 color greyscale
					.backpropType(BackpropType.Standard).build();

			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();

			model.setListeners(new ScoreIterationListener(20));

			System.out.println("TRAIN MODEL");

			FileWriter fileWriter = new FileWriter("src\\main\\resources\\model\\image_classification_model_log.txt");

			for (int i = 0; i < epoch; i++) {
				trainIter.reset();
				while (trainIter.hasNext()) {
					model.fit(trainIter.next());
				}
				Evaluation eval = model.evaluate(testIter);
				System.out.println("\nEpoch: " + i);
				System.out.println(eval.stats());
				fileWriter.write("\nEpoch: " + i + eval.stats());
			}
			fileWriter.close();

			System.out.println("SAVE TRAINED MODEL");

			// Path to save model
			File locationToSave = new File("src\\main\\resources\\model\\image_classification_model.zip");

			boolean saveUpdater = true;

			ModelSerializer.writeModel(model, locationToSave, saveUpdater);
		}
	}
}
