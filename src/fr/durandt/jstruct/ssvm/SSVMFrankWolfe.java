/**
 * 
 */
package fr.durandt.jstruct.ssvm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;


/**
 * <b>Block-Coordinate Frank-Wolfe Optimization for Structural SVMs</b> <br/>
 * Simon Lacoste-Julien, Martin Jaggi, Mark Schmidt, Patrick Pletscher <br/>
 * <i>International Conference on Machine learning (2013)</i>
 * </p>
 * 
 * Solve SSVM with Frank-Wolfe algorithm for convex optimization.
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SSVMFrankWolfe<X,Y> extends SSVM<X,Y> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1837702589626828662L;

	/**
	 * number of iterations
	 */
	protected int maxIter = 5;
	
	protected boolean stochastic = true;

	/**
	 * Train SSVM with Primal-Dual Frank-Wolfe
	 * @param l
	 */
	@Override
	protected void learning(List<STrainingSample<X, Y>> l) {

		// init
		double lk = 0;
		double n = l.size();

		// List of indexes
		List<Integer> index = new ArrayList<Integer>(l.size());
		for(int i=0; i<l.size(); i++) {
			index.add(i,i);
		}

		for(int k=0; k<maxIter; k++) {

			double[] ws = new double[dim];
			double ls = 0;

			if(stochastic) {
				Collections.shuffle(index);
			}

			for(int i : index) {
				//for(int i=0; i<n; i++) {
				STrainingSample<X, Y> ts = l.get(i);

				// loss augmented inference
				Y yp = lossAugmentedInference(ts);

				VectorOp.add(ws, psi(ts.input, ts.output));
				VectorOp.sub(ws, psi(ts.input, yp));

				ls += delta(ts.output, yp);
			}

			ws = VectorOperations.mul(ws, 1/(lambda*n));
			ls /= n;

			double[] diff = VectorOperations.add(w, -1, ws);
			double gamma = (lambda * VectorOperations.dot(diff, w) - lk + ls) / (lambda * VectorOperations.dot(diff, diff));
			gamma = Math.min(Math.max(0, gamma),1);

			VectorOp.add(w, ws, 1-gamma, gamma);
			lk = (1-gamma)*lk + gamma*ls;

			if(verbose>=1) {
				System.out.println("epochs " + k + "/" + maxIter + "\tgamma= " + gamma);
			}
			else if(verbose>=2) {
				System.out.println("epochs " + k + "/" + maxIter + "\tgamma= " + gamma + "\tprimal obj= " + primalObj(l));
			}
			else {
				System.out.print(".");
			}
		}

		if(verbose == 0) {
			System.out.println("*");
		}
	}

	/**
	 * @return the maxIter
	 */
	public int getMaxIter() {
		return maxIter;
	}

	/**
	 * @param maxIter the maxIter to set
	 */
	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}

	@Override
	protected void showParameters() {
		super.showParameters();
		System.out.println("Learning: Frank-Wolfe - maxIter= " + maxIter);
	}

	@Override
	public String toString() {
		String s = "ssvm_FrankWolfe_lambda_" + lambda + "_maxIter_" + maxIter;
		return s;
	}

	/**
	 * @return the stochastic
	 */
	public boolean isStochastic() {
		return stochastic;
	}

	/**
	 * @param stochastic the stochastic to set
	 */
	public void setStochastic(boolean stochastic) {
		this.stochastic = stochastic;
	}
	
	
}
