/**
 * 
 */
package fr.durandt.jstruct.ssvm.segmentation;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationAD3SSVMPegasos extends SSVMPegasos<BagImageSeg,Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5503230296340853689L;

	// List of categories
	protected List<Integer> listClass = null;
	protected int offsetPairwise;

	protected double tradeoff = 1.;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Integer[] lossAugmentedInference(STrainingSample<BagImageSeg, Integer[]> ts, double[] w) {

		int numSuperPixels = ts.input.numberOfInstances();
		int numStates = listClass.size();
		//System.out.println("numSuperPixels= " + numSuperPixels + "\tnumStates= " + numStates);

		if(numSuperPixels == 1) {
			return lossAugmentedInference1(ts,w);
		}

		// Create factor graph.
		FactorGraph factorGraph = new FactorGraph();

		// Create a multi-valued variable for each super-pixel.
		MultiVariable[] multiVariables = new MultiVariable[numSuperPixels];

		for(int i=0; i<numSuperPixels; i++) {
			multiVariables[i] = factorGraph.createMultiVariable(numStates);
			double[] psi = ts.input.getInstance(i);
			for(int k=0; k<numStates; k++) {
				// Assign a log-potential to each state.
				// Ajouter un log ??? normaliser les potentiels ???
				multiVariables[i].setLogPotential(k, valueOf(w, psi, k*psi.length) + delta(ts.output[i], k)); 
			}
		}

		// Design the edge log-potentials.
		List<Double> additionalLogPotentials = new ArrayList<Double>(numStates * numStates);
		int t=0;
		for(int k=0; k<numStates; ++k) {
			for(int l=0; l<numStates; ++l) {
				additionalLogPotentials.add(t, w[k*numStates + l + offsetPairwise]*tradeoff);
				++t;
			}
		}
		//System.out.println("additionalLogPotentials " + additionalLogPotentials);

		// Create a factor for each edge.
		Integer[][] graph = ts.input.getNeigbhors();
		for(int i=0; i<numSuperPixels; ++i) {
			for(int j=0; j<numSuperPixels; ++j) {
				if (graph[i][j] == 1) {
					List<MultiVariable> multiVariablesLocal = new ArrayList<MultiVariable>(2);
					multiVariablesLocal.add(null);
					multiVariablesLocal.add(null);

					multiVariablesLocal.set(0, multiVariables[i]);
					multiVariablesLocal.set(1, multiVariables[j]);
					factorGraph.createFactorDense(multiVariablesLocal, additionalLogPotentials);
				}
			}
		}

		factorGraph.fixMultiVariablesWithoutFactors();
		//factorGraph.print();

		// write file
		/*File file = new File("/Users/thibautdurand/Desktop/These/code_c/AD3/data/java2.uai");
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			bw.write("MARKOV\n");
			bw.write(multiVariables.length + "\n");
			for(int i=0; i<multiVariables.length; i++) {
				bw.write(multiVariables[i].getNumStates() + " ");
			}
			bw.write("\n");
			int nbCliques = numSuperPixels;
			for(int i=0; i<numSuperPixels; ++i) {
				for(int j=0; j<numSuperPixels; ++j) {
					if (graph[i][j] == 1) {
						nbCliques++;
					}
				}
			}
			bw.write(nbCliques + "\n");
			for(int i=0; i<numSuperPixels; ++i) {
				bw.write(1 + " " + i + "\n");
			}
			for(int i=0; i<numSuperPixels; ++i) {
				for(int j=0; j<numSuperPixels; ++j) {
					if (graph[i][j] == 1) {
						bw.write(2 + " " + i + " " + j + "\n");
					}
				}
			}
			bw.write("\n");
			for(int i=0; i<numSuperPixels; ++i) {
				bw.write(numStates + "\n");
				for(int k=0; k<numStates; k++) {
					bw.write(Math.exp(multiVariables[i].getLogPotential(k)) + "\t");
				}
				bw.write("\n");
			}
			for(int i=0; i<numSuperPixels; ++i) {
				for(int j=0; j<numSuperPixels; ++j) {
					if (graph[i][j] == 1) {
						bw.write(additionalLogPotentials.size() + "\n");
						for(t=0; t<additionalLogPotentials.size(); t++) {
							bw.write(Math.exp(additionalLogPotentials.get(t)) + "\t");
						}
						bw.write("\n");
					}
				}
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			e.printStackTrace();
		}*/


		// Run AD3.
		//System.out.println("Running AD3...");
		AD3 solver = new AD3();
		solver.setVerbose(0);
		solver.setEta(0.1);
		solver.setMaxIterations(1000);
		solver.setAdaptEta(true);

		List<Double> posteriors = new ArrayList<Double>();
		List<Double> additionalPosteriors = new ArrayList<Double>();
		double[] value = {0.0};
		solver.solveLPMAP(factorGraph, posteriors, additionalPosteriors, value);

		Integer[] predictLabels = getBestConfiguration(numSuperPixels, numStates, multiVariables, posteriors);

		double v = valueOf(ts.input, predictLabels);
		double delta = delta(ts.output, predictLabels);
		double vi = valueOf(ts.input, ts.output);
		if(v+delta-vi < 0) {
			System.out.println("lai= " + (v+delta) + "\tvi= " + vi + "\tloss= " + (v+delta-vi) + "\tdelta= " + delta);
			System.out.println("gt= " + Arrays.toString(ts.output));
			System.out.println("predictLabels= " + Arrays.toString(predictLabels));
		}
		return predictLabels;

	}

	protected Integer[] lossAugmentedInference1(STrainingSample<BagImageSeg, Integer[]> ts, double[] w) {
		int yp = -1;
		double max = -Double.MAX_VALUE;
		double[] psi = ts.input.getInstance(0);
		for(int i=0; i<listClass.size(); i++) {
			double score = valueOf(w,psi,i*psi.length) + delta(ts.output[0],i);
			if(score > max) {
				max = score;
				yp = i;
			}
		}
		Integer[] ypredict = {yp};
		return ypredict;
	}

	protected double valueOf(double[] w, double[] psi, int offset) {
		double val = 0;
		for(int i=0; i<psi.length; i++) {
			val += w[i+offset] * psi[i];
		}
		return val;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#prediction(java.lang.Object, double[])
	 */
	@Override
	protected Integer[] prediction(BagImageSeg x, double[] w) {

		int numSuperPixels = x.numberOfInstances();
		int numStates = listClass.size();

		if(numSuperPixels == 1) {
			return prediction1(x,w);
		}

		// Create factor graph.
		FactorGraph factorGraph = new FactorGraph();

		// Create a multi-valued variable for each super-pixel.
		MultiVariable[] multiVariables = new MultiVariable[numSuperPixels];

		for(int i=0; i<numSuperPixels; i++) {
			multiVariables[i] = factorGraph.createMultiVariable(numStates);
			double[] psi = x.getInstance(i);
			for(int k=0; k<numStates; k++) {
				// Assign a log-potential to each state.
				// Ajouter un log ???
				multiVariables[i].setLogPotential(k, valueOf(w, psi, k*psi.length)); 
			}
		}

		// Design the edge log-potentials.
		List<Double> additionalLogPotentials = new ArrayList<Double>(numStates * numStates);
		int t=0;
		for(int k=0; k<numStates; ++k) {
			for(int l=0; l<numStates; ++l) {
				additionalLogPotentials.add(t, w[k*numStates + l + offsetPairwise]*tradeoff);
				++t;
			}
		}

		// Create a factor for each edge.
		Integer[][] graph = x.getNeigbhors();
		for(int i=0; i<numSuperPixels; ++i) {
			for(int j=0; j<numSuperPixels; ++j) {
				if (graph[i][j] == 1) {
					List<MultiVariable> multiVariablesLocal = new ArrayList<MultiVariable>(2);
					multiVariablesLocal.add(null);
					multiVariablesLocal.add(null);

					multiVariablesLocal.set(0, multiVariables[i]);
					multiVariablesLocal.set(1, multiVariables[j]);
					factorGraph.createFactorDense(multiVariablesLocal, additionalLogPotentials);
				}
			}
		}

		factorGraph.fixMultiVariablesWithoutFactors();

		// Run AD3.
		//System.out.println("Running AD3...");
		AD3 solver = new AD3();
		solver.setVerbose(0);
		solver.setEta(0.1);
		solver.setMaxIterations(1000);
		solver.setAdaptEta(true);

		List<Double> posteriors = new ArrayList<Double>();
		List<Double> additionalPosteriors = new ArrayList<Double>();
		double[] value = {0.0};
		solver.solveLPMAP(factorGraph, posteriors, additionalPosteriors, value);

		Integer[] predictLabels = getBestConfiguration(numSuperPixels, numStates, multiVariables, posteriors);
		return predictLabels;
	}

	protected Integer[] prediction1(BagImageSeg x, double[] w) {
		int yp = -1;
		double max = -Double.MAX_VALUE;
		double[] psi = x.getInstance(0);
		for(int i=0; i<listClass.size(); i++) {
			double score = valueOf(w,psi,i*psi.length);
			if(score > max) {
				max = score;
				yp = i;
			}
		}
		Integer[] ypredict = {yp};
		return ypredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#evaluation(java.util.List)
	 */
	public double accuracySuperpixels(List<STrainingSample<BagImageSeg, Integer[]>> l) {
		return empiricalRisk(l);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#delta(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(Integer[] yi, Integer[] y) {
		double val = 0;
		for(int i=0; i<yi.length; i++) {
			val += delta(yi[i], y[i]);
		}
		return val;
	}

	protected double delta(Integer yi, Integer y) {
		if(y == yi) {
			return 0;
		}
		else {
			return 1;
		}
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImageSeg x, Integer[] y) {
		double[] psi = new double[dim];

		// Data term
		for(int i=0; i<y.length; i++) {
			double[] psiy = x.getInstance(i);
			int offset = y[i] * psiy.length;
			for(int d=0; d<psiy.length; d++) {
				psi[d + offset] += psiy[d];
			}
		}

		// Pairwise term
		Integer[][] graph = x.getNeigbhors();
		int nbSuperpixels = graph.length;
		int nbLabels = listClass.size();
		for(int i=0; i<nbSuperpixels; i++) {
			for(int j=0; j<nbSuperpixels; j++) {
				if(graph[i][j] == 1) {
					psi[y[i]*nbLabels+y[j]+offsetPairwise] += tradeoff;
				}
			}
		}

		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<BagImageSeg, Integer[]>> l) {

		int nbClass = 0;
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			nbClass = Math.max(nbClass, VectorOp.max(ts.output));
		}
		nbClass++;

		int[] numberOfSuperpixels = new int[nbClass];
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			for(int y : ts.output) {
				numberOfSuperpixels[y]++;
			}
		}

		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Compute the dimension of the unary 
		offsetPairwise = listClass.size()*l.get(0).input.getInstance(0).length;

		// Compute the dimension of w
		dim = listClass.size()*l.get(0).input.getInstance(0).length + listClass.size()*listClass.size();

		// Initialize w
		w = new double[dim];
		for(int i=0; i<dim; i++) {
			w[i] = 0.0;//1.0;
		}

		System.out.println("Segmentation SSVM with AD3 - classes: " + listClass + "\toffset= " + offsetPairwise);
		System.out.println(Arrays.toString(numberOfSuperpixels));

	}

	protected void print(double[][] mat) {
		for(int i=0; i<mat.length; i++) {
			for(int j=0; j<mat[i].length; j++) {
				System.out.print(mat[i][j] + "\t");
			}
			System.out.println();
		}
	}

	protected Integer[] getBestConfiguration(int numNodes, int numStates, MultiVariable[] multiVariables, List<Double> posteriors) {
		int offset=0;
		Integer[] bestStates = new Integer[numNodes];
		for(int i=0; i<numNodes; ++i) {
			int best = -1;
			for(int k = 0; k < numStates; ++k) {
				if (best < 0 || posteriors.get(offset+k) > posteriors.get(offset+best)) {
					best = k;
				}
			}
			offset += numStates;
			bestStates[i] = best;
		}
		//System.out.println("bestStates= " + Arrays.toString(bestStates));
		return bestStates;
	}

	public double evaluation(List<STrainingSample<BagImageSeg, Integer[]>> l) {
		double loss = 0;
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			// Compute the prediction
			Integer[] yp = prediction(ts.input, w);
			loss += delta(ts.output, yp);
		}
		loss /= l.size();
		return loss;
	}

	public double evaluationPixelAccuracy(List<STrainingSample<BagImageSeg, Integer[]>> l) {
		// The number of correct classify pixels
		long correct = 0;
		// The number of pixels
		long allPixels = 0;
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			// Compute the prediction
			Integer[] yp = prediction(ts.input, w);
			// Get the predicted mask
			BufferedImage predictedMask = ts.input.predictedMask(yp);
			// Get the ground truth mask
			BufferedImage maskGT = ts.input.getGtMask();

			for(int i=0; i<predictedMask.getHeight(); i++) {
				for(int j=0; j<predictedMask.getWidth(); j++) {
					// Get the value of predicted mask pixel[i,j] (i.e the class of the pixel)
					int pixel = predictedMask.getRGB(j, i);
					int predictLabel = (pixel) & 0xff;
					// Get the value of ground truth mask pixel[i,j] (i.e the class of the pixel)
					pixel = maskGT.getRGB(j, i);
					int gtLabel = (pixel) & 0xff;
					// Test if the class of the pixel is the same for the ground truth and the predicted mask
					if(gtLabel == predictLabel) {
						// Count the number of correct classify pixels
						correct++;
					}
					// Count the number of pixels
					allPixels++;
				}
			}	
		}
		// Compute the accuracy
		double accuracy = (double)correct / (double)allPixels;
		System.out.println("pixel accuracy= " + accuracy + "\t( " + correct + " / " + allPixels + " )");
		return accuracy;
	}

	public double evaluationPerClass(List<STrainingSample<BagImageSeg, Integer[]>> l) {

		// Compute the confusion matrix
		long[][] confusion = new long[listClass.size()][listClass.size()];
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			Integer[] yp = prediction(ts.input, w);
			BufferedImage predictedMask = ts.input.predictedMask(yp);
			BufferedImage maskGT = ts.input.getGtMask();
			for(int i=0; i<predictedMask.getHeight(); i++) {
				for(int j=0; j<predictedMask.getWidth(); j++) {
					int pixel = predictedMask.getRGB(j, i);
					int predictLabel = (pixel) & 0xff;
					pixel = maskGT.getRGB(j, i);
					int gtLabel = (pixel) & 0xff;
					if(gtLabel != 0) {
						confusion[gtLabel-1][predictLabel-1]++;
					}
				}
			}	
		}

		// Compute G
		long[] g = new long[listClass.size()];
		for(int i=0; i<listClass.size(); i++) {
			for(int j=0; j<listClass.size(); j++) {
				g[i] += confusion[i][j];
			}
		}

		// Compute the per-class accuracy
		double pc = 0;
		for(int i=0; i<listClass.size(); i++) {
			if(g[i] != 0) {
				pc += (double)confusion[i][i] / (double)g[i];
			}
		}
		pc /= listClass.size();

		System.out.println("per class accuracy= " + pc);
		return pc;
	}
}
