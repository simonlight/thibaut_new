/**
 * 
 */
package fr.durandt.jstruct.ssvm.segmentation;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMCuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationMulticlassSSVMCuttingPlane1Slack extends SSVMCuttingPlane1Slack<BagImageSeg,Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6678350146462582408L;
	
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
			// Gets the representation of (i+1)th superpixel 
			double[] psi = ts.input.getInstance(i);
			Integer ypredict = -1;
			double valMax = -Double.MAX_VALUE;
			// Search the class which maximize the score
			for(int y : listClass) {
				double val = valueOf(w, psi, y*psi.length) + delta(ts.output[i], y);
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
	 * 
	 * @param w
	 * @param psi
	 * @param offset
	 * @return
	 */
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

		int nbSuperPixels = x.numberOfInstances();
		
		// Infer the label of each superpixel independently of other superpixels
		Integer[] predictLabels = new Integer[nbSuperPixels];
		for(int i=0; i<nbSuperPixels; i++) {
			double[] psi = x.getInstance(i);
			Integer ypredict = -1;
			double valMax = -Double.MAX_VALUE;
			for(int y : listClass) {
				double val = valueOf(w, psi, y*psi.length);
				if(val > valMax) {
					ypredict = y;
					valMax = val;
				}
			}
			predictLabels[i] = ypredict;
		}
		return predictLabels;
	}

	public double evaluation(List<STrainingSample<BagImageSeg, Integer[]>> l) {
		double loss = 0;
		for(STrainingSample<BagImageSeg, Integer[]> ts : l) {
			Integer[] yp = prediction(ts.input, w);
			loss += delta(ts.output, yp);
		}
		loss /= l.size();
		return loss;
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

		System.out.println("Segmentation SSVM - classes: " + listClass);

	}

	protected void print(double[][] mat) {
		for(int i=0; i<mat.length; i++) {
			for(int j=0; j<mat[i].length; j++) {
				System.out.print(mat[i][j] + "\t");
			}
			System.out.println();
		}
	}
}
