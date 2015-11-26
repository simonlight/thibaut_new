/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.multiclass;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LatentML3PegasosBagImage extends LatentML3Pegasos<BagImage, Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5535675051985623621L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.multiclass.LatentML3#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.multiclass.LatentML3#prediction(java.lang.Object, java.lang.Integer, double[][][])
	 */
	@Override
	protected Integer prediction(BagImage x, Integer y, double[][][] w) {
		double max = -Double.MAX_VALUE;
		int hpredict = -1;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double score = valueOf(x,y,h,w);
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.multiclass.LatentML3#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImage, Integer>, Integer>> l) {

		// Initialize the dimension of the features
		d = l.get(0).input.x.getInstance(0).length;

		// Search the number of classes
		c = 0;
		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ts : l) {
			c = Math.max(c, ts.output);
		}
		c++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<c; i++) {
			listClass.add(i);
		}

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.LatentStructuralClassifier#predictionOutputLatent(java.lang.Object)
	 */
	@Override
	public Object[] predictionOutputLatent(BagImage x) {
		int ypredict = -1;	// class prediction
		Integer hpredict = null;	// latent prediction
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			for(int h=0; h<x.numberOfInstances(); h++) {	// For each latent
				// Compute the score for a given class y and region h
				double val = valueOf(x, y, h, w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

}
