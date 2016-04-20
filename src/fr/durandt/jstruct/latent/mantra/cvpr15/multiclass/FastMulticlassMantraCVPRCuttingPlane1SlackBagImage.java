/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.cvpr15.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class FastMulticlassMantraCVPRCuttingPlane1SlackBagImage extends FastMulticlassMantraCVPRCuttingPlane1Slack<BagImage, Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9086214837120320127L;

	private int initType = 0;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#delta(java.lang.Integer, java.lang.Integer, java.lang.Object)
	 */
	@Override
	protected double delta(Integer y, Integer yp, Integer hp) {
		if(y == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImage, Integer>, Integer>> l) {

		// Count the number of classes
		int nbClass = 0;	// Number of classes
		for(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;

		// Create the list of classes
		listClass = new ArrayList<Integer>();	// List of classes
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Count the number of samples per class
		double[] nb = new double[nbClass];
		for(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts : l) {
			nb[ts.output]++;
		}

		// Print the classes and the number of samples per class
		System.out.println("Multiclass MANTRA - classes: " + listClass + "\t" + Arrays.toString(nb));

		// Define the dimension of w
		dim = l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[listClass.size()][dim];

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#prediction(java.lang.Object, java.lang.Integer, double[][])
	 */
	@Override
	protected Integer prediction(BagImage x, Integer y, double[][] w) {
		Integer hpredict = null;
		double max = -Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] psi = psi(x, h);
			double val = linear.valueOf(w[y], psi);
			if(val > max){
				max = val;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#predictionOutputLatent(java.lang.Object, double[][])
	 */
	@Override
	protected Object[] predictionOutputLatent(BagImage x, double[][] w) {
		ComputedScoresMinMax<Integer> precomputedScore = precomputedScores(x, w);
		double max = -Double.MAX_VALUE;
		Integer yp = -1;
		for(Integer y :listClass) {
			double score = valueOf(x, y, precomputedScore);
			if(score > max) {
				max = score;
				yp = y;
			}
		}
		Object[] res = new Object[2];
		res[0] = yp;
		res[1] = precomputedScore.getHmax(yp);
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[][])
	 */
	@Override
	protected Integer prediction(LatentRepresentation<BagImage, Integer> x, double[][] w) {
		return (Integer) predictionOutputLatent(x.x, w)[0];
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#valueOfHPlusMoins(fr.durandt.jstruct.latent.LatentRepresentation, int, double[][])
	 */
	@Override
	protected Object[] valueOfHPlusMinus(BagImage x, int y, double[][] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] psi = psi(x, h);
			double val = linear.valueOf(psi, w[y]);
			if(val>valmax){
				valmax = val;
				hmax = h;
			}
			if(val<valmin){
				valmin = val;
				hmin = h;
			}
		}
		Object[] res = new Object[4];
		res[0] = hmax;
		res[1] = valmax;
		res[2] = hmin;
		res[3] = valmin;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#initLatent(fr.durandt.jstruct.struct.STrainingSample)
	 */
	@Override
	protected Integer[] initLatent(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts) {
		Integer[] hinit = new Integer[2];
		if(initType == 0) {
			hinit[0] = 0;
			hinit[1] = 0;//ts.sample.x.getFeatures().size()-1;
		}
		else if(initType == 1) {
			hinit[0] = (int)(Math.random()*ts.input.x.numberOfInstances());
			hinit[1] = (int)(Math.random()*ts.input.x.numberOfInstances());
		}
		else if(initType == 2) {
			hinit[0] = 0;
			hinit[1] = (int)(Math.random()*ts.input.x.numberOfInstances());
		}
		else {
			System.out.println("error init");
			System.exit(0);
		}
		return hinit;
	}
	
	/**
	 * Compute the multi-class accuracy
	 * @param l
	 * @return
	 */
	public double accuracy(List<STrainingSample<LatentRepresentation<BagImage, Integer>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

}
