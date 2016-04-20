/**
 * 
 */
package fr.durandt.jstruct.ssvm.segmentation;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationMulticlassSSVMPegasos extends SSVMPegasos<BagImageSeg,Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4490990326181272989L;

	// List of categories
	protected List<Integer> listClass = null;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Integer[] lossAugmentedInference(STrainingSample<BagImageSeg, Integer[]> ts, double[] w) {

		int nbSuperPixels = ts.input.numberOfInstances();

		// Infer the label of each superpixel independently of other superpixels
		Integer[] predictLabels = new Integer[nbSuperPixels];
		for(int i=0; i<nbSuperPixels; i++) {
			// Get the representation of (i+1)th superpixel 
			double[] psi = ts.input.getInstance(i);
			Integer ypredict = -1;
			double valMax = -Double.MAX_VALUE;
			// Predict the label for the (i+1)th superpixel: search the class which maximize the score
			for(int y : listClass) {
				double val = valueOf(w, psi, y) + delta(ts.output[i], y);
				if(val > valMax) {
					ypredict = y;
					valMax = val;
				}
			}
			predictLabels[i] = ypredict;
		}

		return predictLabels;
	}

	/**
	 * Compute the score for a given class &lt w_y, psi &gt
	 * @param w
	 * @param psi
	 * @param y
	 * @return &lt w_y, psi &gt
	 */
	protected double valueOf(double[] w, double[] psi, int y) {
		int offset = y*psi.length;
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
		// Number of superpixels
		int nbSuperPixels = x.numberOfInstances();

		// Infer the label of each superpixel independently of other superpixels
		Integer[] predictLabels = new Integer[nbSuperPixels];
		for(int i=0; i<nbSuperPixels; i++) {
			// Get the representation of (i+1)th superpixel 
			double[] psi = x.getInstance(i);
			// Predict the label for the (i+1)th superpixel: the class with the maximum score
			Integer ypredict = -1;
			double valMax = -Double.MAX_VALUE;
			for(int y : listClass) {
				// Compute the score for class y
				double val = valueOf(w, psi, y);
				if(val > valMax) {
					ypredict = y;
					valMax = val;
				}
			}
			predictLabels[i] = ypredict;
		}
		// Return predicted labels for each superpixel
		return predictLabels;
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

		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Compute the dimension of w
		dim = listClass.size()*l.get(0).input.getInstance(0).length;

		// Initialize w
		w = new double[dim];

		System.out.println("Segmentation Multiclass SSVM - classes: " + listClass);

	}

	@Override
	public String toString() {
		String s = "segmentation_multiclass_" + super.toString();
		return s;
	}
}
