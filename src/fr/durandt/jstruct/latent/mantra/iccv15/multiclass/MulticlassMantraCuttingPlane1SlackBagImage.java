/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.mantra.iccv15.MantraCuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class MulticlassMantraCuttingPlane1SlackBagImage extends MantraCuttingPlane1Slack<BagImage, Integer, Integer> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = -3627089131898247850L;

	private int initType = 0;

	/**
	 * List of classes {0,1,...,c-1}
	 */
	protected List<Integer> listClass = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImage x, Integer y, Integer h) {
		double[] psi = new double[dim];
		int offset = y*x.getInstance(h).length;
		for(int i=0; i<x.getInstance(h).length; i++) {
			psi[offset + i] = x.getInstance(h)[i];
		}
		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#delta(java.lang.Integer, java.lang.Integer, java.lang.Object)
	 */
	@Override
	protected double delta(Integer y, Integer yp, Integer hp) {
		if(y == yp) {
			return 0.;
		}
		else {
			return 1.;
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
		System.out.println("Multiclass SSVM - classes: " + listClass + "\t" + Arrays.toString(nb));

		// Define the dimension of w
		dim = l.get(0).input.x.getInstance(0).length*listClass.size();

		// Initialize w
		w = new double[dim];
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#prediction(java.lang.Object, java.lang.Integer, double[][])
	 */
	@Override
	protected Object[] prediction(BagImage x, Integer y, double[] w) {
		Object[] or = valueOfHPlusMinus(x, y, w);
		Object[] res = new Object[2];
		res[0] = or[0];
		res[1] = or[2];
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#predictionOutputLatent(java.lang.Object, double[][])
	 */
	@Override
	protected Object[] predictionOutputLatent(BagImage x, double[] w) {
		int ypredict = -1;	// class prediction
		Integer hppredict = null;	// latent prediction h^+
		Integer hmpredict = null;	// latent prediction h^-
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			Object[] or = valueOfHPlusMinus(x, y, w);
			double vmax = (double) or[1];
			double vmin = (double) or[3];
			double val = vmax + vmin;
			if(val>valmax){
				valmax = val;
				ypredict = y;
				hppredict = (Integer) or[0];
				hmpredict = (Integer) or[2];
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hppredict;
		res[2] = hmpredict;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPR#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[][])
	 */
	@Override
	protected Integer prediction(LatentRepresentation<BagImage, Integer> x, double[] w) {
		return (Integer) predictionOutputLatent(x.x, w)[0];
	}

	protected Object[] valueOfHPlusMinus(BagImage x, int y, double[] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] psi = psi(x, y, h);
			double val = linear.valueOf(psi, w);
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

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.Mantra#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<BagImage, Integer>, Integer> ts, double[] w) {
		int ypredict = -1;	// class prediction
		Integer hppredict = null;	// latent prediction h^+
		Integer hmpredict = null;	// latent prediction h^-
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			Object[] or = valueOfHPlusMinus(ts.input.x, y, w);
			double vmax = (double) or[1];
			double vmin = (double) or[3];
			double val = vmax + vmin + delta(ts.output, y, (Integer) or[0]);
			if(val>valmax){
				valmax = val;
				ypredict = y;
				hppredict = (Integer) or[0];
				hmpredict = (Integer) or[2];
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hppredict;
		res[2] = hmpredict;
		return res;
	}

}
