/**
 * 
 */
package fr.durandt.jstruct.ssvm.segmentation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jstruct.extern.lib.gco.GCO;
import fr.durandt.jstruct.ssvm.SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationGraphCutSSVMPegasos extends SSVMPegasos<BagImageSeg,Integer[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5134409536002332018L;

	// List of categories
	protected List<Integer> listClass = null;
	protected int offsetPairwise;

	protected GCO gco = null;
	protected int grapthCutOptim = 0;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Integer[] lossAugmentedInference(STrainingSample<BagImageSeg, Integer[]> ts, double[] w) {

		int nbSuperPixels = ts.input.numberOfInstances();
		int nbLabels = listClass.size();

		if(nbSuperPixels == 1) {
			return lossAugmentedInference1(ts,w);
		}

		// unary term
		double[] unary = new double[nbLabels * nbSuperPixels];
		for(int p=0; p<nbSuperPixels; p++) {
			double[] psi = ts.input.getInstance(p);
			for(int l=0; l<nbLabels; l++) {
				unary[p*nbLabels + l] = -valueOf(w, psi, l*psi.length) - delta(ts.output[p], l);
			}
		}
		if(verbose > 1) {
			System.out.println("gt= " + Arrays.toString(ts.output));
		}
		if(verbose > 2) {
			System.out.println("unary=");
			for(int i=0; i<nbLabels; i++) {
				for(int j=0; j<nbSuperPixels; j++) {
					System.out.print(unary[i*nbSuperPixels + j] + "\t");
				}
				System.out.println();
			}
		}

		// Structure du graph
		Integer[][] graph = ts.input.getNeigbhors();
		int[] neighbors = new int[nbSuperPixels * nbSuperPixels];
		for(int p=0; p<nbSuperPixels; p++) {
			for(int q=0; q<nbSuperPixels; q++) {
				neighbors[p*nbSuperPixels + q] = (int)graph[p][q];
			}
		}

		double[] pairwise = new double[nbLabels * nbLabels];
		for(int l1=0; l1<nbLabels; l1++) {
			for(int l2=0; l2<nbLabels; l2++) {
				pairwise[l1 + nbLabels*l2] = -w[l1*nbLabels + l2 + offsetPairwise];
			}
		}

		if(verbose > 2) {
			System.out.println("pairwise");
			for(int i=0; i<nbLabels; i++) {
				for(int j=0; j<nbLabels; j++) {
					System.out.print(pairwise[i*nbLabels + j] + "\t");
				}
				System.out.println();
			}
		}

		int[] labels = new int[nbSuperPixels];

		gco.getGcolib().DoubleGeneralGraph(nbSuperPixels, nbLabels, unary, pairwise, neighbors, labels, grapthCutOptim, 0);

		if(verbose > 1) {
			System.out.println("LAI output= " + Arrays.toString(labels));
		}

		Integer[] predictLabels = new Integer[labels.length];
		for(int i=0; i<labels.length; i++) {
			predictLabels[i] = labels[i];
		}

		double v = valueOf(ts.input, predictLabels);
		double delta = delta(ts.output, predictLabels);
		double vi = valueOf(ts.input, ts.output);
		//System.out.println("lai= " + (v+delta) + "\tvi= " + vi + "\tloss= " + (v+delta-vi));
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

		int nbSuperPixels = x.numberOfInstances();
		int nbLabels = listClass.size();

		if(nbSuperPixels == 1) {
			return prediction1(x,w);
		}

		// unary term
		double[] unary = new double[nbLabels * nbSuperPixels];
		for(int p=0; p<nbSuperPixels; p++) {
			double[] psi = x.getInstance(p);
			for(int l=0; l<nbLabels; l++) {
				unary[p*nbLabels + l] = -valueOf(w, psi, l*psi.length);
			}
		}

		if(verbose > 2) {
			System.out.println("unary=");
			for(int i=0; i<nbLabels; i++) {
				for(int j=0; j<nbSuperPixels; j++) {
					System.out.print(unary[i*nbSuperPixels + j] + "\t");
				}
				System.out.println();
			}
		}

		// Structure du graph
		Integer[][] graph = x.getNeigbhors();
		int[] neighbors = new int[nbSuperPixels * nbSuperPixels];
		for(int p=0; p<nbSuperPixels; p++) {
			for(int q=0; q<nbSuperPixels; q++) {
				neighbors[p*nbSuperPixels + q] = (int)graph[p][q];
			}
		}

		double[] pairwise = new double[nbLabels * nbLabels];
		for(int l1=0; l1<nbLabels; l1++) {
			for(int l2=0; l2<nbLabels; l2++) {
				pairwise[l1 + nbLabels*l2] = -w[l1*nbLabels + l2 + offsetPairwise];
			}
		}

		int[] labels = new int[nbSuperPixels];

		gco.getGcolib().DoubleGeneralGraph(nbSuperPixels, nbLabels, unary, pairwise, neighbors, labels, grapthCutOptim, 0);

		if(verbose > 1) {
			System.out.println("labels= " + Arrays.toString(labels));
		}

		Integer[] predictLabels = new Integer[labels.length];
		for(int i=0; i<labels.length; i++) {
			predictLabels[i] = labels[i];
		}
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
			for(int j=i; j<nbSuperpixels; j++) {
				if(graph[i][j] == 1) {
					psi[y[i]*nbLabels+y[j]+offsetPairwise] = 1;
					psi[y[j]*nbLabels+y[i]+offsetPairwise] = 1;
				}
				else {
					psi[y[i]*nbLabels+y[j]+offsetPairwise] = 0;
					psi[y[j]*nbLabels+y[i]+offsetPairwise] = 0;
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
			w[i] = 0;
		}

		System.out.println("Segmentation SSVM - classes: " + listClass + "\toffset= " + offsetPairwise);
		System.out.println(Arrays.toString(numberOfSuperpixels));

		gco = new GCO();

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
