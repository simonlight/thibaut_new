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
public class MulticlassMantraCVPRCuttingPlane1SlackBagImage extends MulticlassMantraCVPRCuttingPlane1Slack<BagImage, Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 199633184341219833L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.MulticlassMantraCVPRCuttingPlane1Slack#valueOfHPlusMinus(java.lang.Object, int, double[])
	 */
	@Override
	protected Object[] valueOfHPlusMinus(BagImage x, int y, double[] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] psi = psi(x, y, h);
			double val = linear.valueOf(w, psi);
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
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImage x, Integer y, Integer h) {
		double[] psi = new double[dim];
		double[] xx = x.getInstance(h);
		for(int i=0; i<xx.length; i++) {
			psi[y*xx.length+i] = xx[i];
		}
		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#delta(java.lang.Object, java.lang.Object, java.lang.Object)
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
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#init(java.util.List)
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
		dim = listClass.size()*l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[dim];

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Integer prediction(BagImage x, Integer y, double[] w) {
		Integer hpredict = null;
		double max = -Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] psi = psi(x, y, h);
			double val = linear.valueOf(w, psi);
			if(val > max){
				max = val;
				hpredict = h;
			}
		}
		return hpredict;
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
