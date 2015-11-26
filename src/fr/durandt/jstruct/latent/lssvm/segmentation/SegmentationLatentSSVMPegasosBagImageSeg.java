/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.segmentation;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationLatentSSVMPegasosBagImageSeg extends SegmentationLatentSSVMPegasos<BagImageSeg,Integer[],Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -836176176995211868L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.segmentation.SegmentationLatentSSVM#deltaC(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double deltaC(Integer[] yi, Integer[] hi) {
		
		double delta = 0;
		for(int y=0; y<yi.length; y++) {
			if(yi[y] == 1) {
				delta += deltaC(y,hi);
			}
		}
		return delta;
	}
	
	protected double deltaC(Integer yi, Integer[] hi) {
		for(int i=0; i<hi.length; i++) {
			if(hi[i] == yi) {
				return 0.;
			}
		}
		return 1.;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImageSeg x, Integer[] y, Integer[] h) {
		double[] psi = new double[dim];

		for(int i=0; i<h.length; i++) {
			double[] psih = x.getInstance(i);
			int offset = h[i] * psih.length;
			for(int d=0; d<psih.length; d++) {
				psi[d + offset] += psih[d];
			}
		}

		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(Integer[] y, Integer[] yp, Integer[] hp) {
		double delta = 0;
		for(int i=0; i<hp.length; i++) {
			delta += delta(y,hp[i]);
		}
		return delta;
	}

	protected double delta(Integer[] y, Integer h) {
		if(y[h] == 1) {
			return 0.;
		}
		else {
			return 1.;
		}
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]>> l) {
		
		int nbClass = l.get(0).output.length;

		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Compute the dimension of w
		dim = listClass.size()*l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[dim];

		System.out.println("Segmentation Multiclass Latent SSVM - classes: " + listClass);

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]> ts, double[] w) {
		// Initialize the predicted output
		Integer[] predictedOutput = new Integer[listClass.size()];
		Integer[] predictedLatent = new Integer[ts.input.x.numberOfInstances()];

		for(int j=0; j<ts.input.x.numberOfInstances(); j++) {
			// Compute the loss augmented inference for super-pixel j
			predictedLatent[j] = lossAugmentedInference(ts, j, w);
			// Update the predicted output 
			predictedOutput[predictedLatent[j]] = 1;
		}

		Object[] res = new Object[2];
		res[0] = predictedOutput;
		res[1] = predictedLatent;

		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Integer[] prediction(BagImageSeg x, Integer[] y, double[] w) {

		List<Integer> listClassRest = new ArrayList<Integer>();
		for(int i=0; i<y.length; i++) {
			if(y[i] == 1) {
				listClassRest.add(i);
			}
		}

		// Compute the scores for
		double[][] scores = new double[x.numberOfInstances()][listClassRest.size()];
		double[][] scoresCopy = new double[x.numberOfInstances()][listClassRest.size()];
		for(int j=0; j<x.numberOfInstances(); j++) {
			double[] psi = x.getInstance(j);
			for(int i=0; i<listClassRest.size(); i++) {
				int yy = listClassRest.get(i);
				scores[j][i] = valueOf(w, psi, yy);
				scoresCopy[j][i] = scores[j][i];
			}
		}

		// affecte une région à chaque classe présente
		Integer[] latent = new Integer[x.numberOfInstances()];
		for(int i=0; i<listClassRest.size(); i++) {
			Integer[] res = getMax(scores);
			int pmax = res[0];
			int ymax = res[1];
			latent[pmax] = listClassRest.get(ymax);

			scores[pmax] = null;
			for(int p=0; p<x.numberOfInstances(); p++) {
				if(scores[p] != null) {
					scores[p][ymax] = -Double.NaN;
				}
			}
		}

		// prédit la classe pour les super-pixels sans classes
		for(int j=0; j<x.numberOfInstances(); j++) {
			if(scores[j] != null) {
				latent[j] = getMax(scoresCopy[j]);
			}
		}
		
		//System.out.println("yi= " + Arrays.toString(y));
		//System.out.println("hi= " + Arrays.toString(latent));
		
		return latent;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	/*@Override
	protected Integer[] prediction(BagImageSeg x, Integer[] y, double[] w) {

		List<Integer> listClassRest = new ArrayList<Integer>();
		for(int i=0; i<y.length; i++) {
			if(y[i] == 1) {
				listClassRest.add(i);
			}
		}

		// Compute the scores for
		double[][] scores = new double[x.numberOfInstances()][listClassRest.size()];
		for(int j=0; j<x.numberOfInstances(); j++) {
			double[] psi = x.getInstance(j);
			for(int i=0; i<listClassRest.size(); i++) {
				int yy = listClassRest.get(i);
				scores[j][i] = valueOf(w, psi, yy);
			}
		}

		Integer[] latent = new Integer[x.numberOfInstances()];
		// prédit la classe pour les super-pixels sans classes
		for(int j=0; j<x.numberOfInstances(); j++) {
			if(scores[j] != null) {
				latent[j] = getMax(scores[j]);
			}
		}
		
		//System.out.println("yi= " + Arrays.toString(y));
		//System.out.println("hi= " + Arrays.toString(latent));
		
		return latent;
	}*/
	
	
	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(BagImageSeg x, double[] w) {

		// Initialize the predicted output
		Integer[] predictedOutput = new Integer[listClass.size()];
		Integer[] predictedLatent = new Integer[x.numberOfInstances()];

		for(int j=0; j<x.numberOfInstances(); j++) {
			// Compute the latent prediction
			predictedLatent[j] = prediction(x, j, w);
			// update the predicted output 
			predictedOutput[predictedLatent[j]] = 1;
		}

		Object[] res = new Object[2];
		res[0] = predictedOutput;
		res[1] = predictedLatent;

		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected Integer[] prediction(LatentRepresentation<BagImageSeg, Integer[]> x, double[] w) {
		Object[] res = predictionOutputLatent(x.x, w);
		return (Integer[]) res[0];
	}

	/**
	 * Compute the latent prediction for super-pixel j with given model w
	 * @param x
	 * @param j
	 * @param w
	 * @return
	 */
	protected Integer prediction(BagImageSeg x, int j, double[] w) {

		double[] psi = x.getInstance(j);

		int ypredict = -1;
		double max = -Double.MAX_VALUE;

		for(int y : listClass) {
			double score = valueOf(w, psi, y);
			if(score > max) {
				max = score;
				ypredict = y;
			}
		}

		return ypredict;
	}

	protected Integer lossAugmentedInference(STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]> ts, int j, double[] w) {

		double[] psi = ts.input.x.getInstance(j);

		int hpredict = -1;
		double max = -Double.MAX_VALUE;

		for(int h : listClass) {
			double score = valueOf(w, psi, h) + delta(ts.output, h);
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}

		return hpredict;
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

	protected Integer[] getMax(double[][] scores) {

		double max = -Double.MAX_VALUE;
		int ymax = -1;
		int pmax = -1;

		for(int p=0; p<scores.length; p++) {
			if(scores[p] != null) {
				for(int y=0; y<scores[p].length; y++) {
					if(!Double.isNaN(scores[p][y]) && scores[p][y] > max) {
						max = scores[p][y];
						pmax = p;
						ymax = y;
					}
				}
			}
		}

		Integer[] res = {pmax, ymax};
		return res;
	}

	protected Integer getMax(double[] scores) {

		double max = -Double.MAX_VALUE;
		int ymax = -1;

		for(int y=0; y<scores.length; y++) {
			if(!Double.isNaN(scores[y]) && scores[y] > max) {
				max = scores[y];
				ymax = y;
			}
		}

		return ymax;
	}
	
	public double evaluationPerClass(List<STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]>> l) {

		// Compute the confusion matrix
		long[][] confusion = new long[listClass.size()][listClass.size()];
		for(STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]> ts : l) {
			Object[] res = predictionOutputLatent(ts.input.x, w);
			Integer[] yp = (Integer[]) res[1]; 
			BufferedImage predictedMask = ts.input.x.predictedMask(yp);
			BufferedImage maskGT = ts.input.x.getGtMask();
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
