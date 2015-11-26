/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.segmentation;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.LSSVM;
import fr.durandt.jstruct.struct.STrainingSample;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SegmentationLatentSSVM<X,Y,H> extends LSSVM<X,Y,H> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 3910669332958451270L;

	/**
	 * list of classes {0,1,...,c-1}
	 */
	protected List<Integer> listClass = null;


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Abstract methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Loss function 
	 * @param yi
	 * @param hi
	 * @return
	 */
	protected abstract double deltaC(Y yi, H hi);


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Compute the loss of the objective function <br />
	 * \sum_{i=1}^N max_{y,h} ( delta(yi,y) + &lt w, \psi(xi,y,h) &gt ) - max_hi ( &lt w,psi(xi,yi,hp) &gt + deltaC(yi,hi) )
	 * @param l list of training samples
	 */
	@Override
	protected double loss(List<STrainingSample<LatentRepresentation<X,H>,Y>> l, double[] w) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Y> ts : l) {
			// Compute the loss augmented inference
			Object[] or = lossAugmentedInference(ts, w);
			Y yp = (Y)or[0];
			H hp = (H)or[1];
			// Compute the "best" latent value for ground truth output
			H h = prediction(ts.input.x, ts.output, w);
			// Compute the loss with the predicted output and latent variables
			loss += delta(ts.output, yp, hp) + valueOf(ts.input.x, yp, hp) - valueOf(ts.input.x, ts.output, h) + deltaC(ts.output, h);

		}
		loss /= l.size();
		return loss;	
	}

	/**
	 * Print the parameters
	 */
	@Override
	protected void showParameters(){
		System.out.println("Train LSSVM Segmentation \tlambda: " + lambda + "\tdim: " + w.length);
	}
}
