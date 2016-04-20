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
 * Solve SSVM with a randomized block-coordinate variant of the classic 
 * Frank-Wolfe algorithm for convex optimization with block-separable constraints.
 * </p>
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class SSVMFrankWolfeBlockCoordinate<X,Y> extends SSVM<X,Y> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1837702589626828662L;

	protected int maxIter = 50;

	/**
	 * Train SSVM with Block-Coordinate Primal-Dual Frank-Wolfe
	 * @param l
	 */
	@Override
	protected void learning(List<STrainingSample<X, Y>> l) {

		// init
		double[][] wall = new double[l.size()][dim];
		double[] lall = new double[l.size()];

		// List of indexes
		List<Integer> index = new ArrayList<Integer>(l.size());
		for(int i=0; i<l.size(); i++) {
			index.add(i,i);
		}

		for(int k=0; k<maxIter; k++) {
			// select randomly a sample
			//int i = (int)(Math.random()*l.size());
			Collections.shuffle(index);
			
			for(int i : index) {
				STrainingSample<X, Y> ts = l.get(i);

				// loss augmented inference
				Y ystar = lossAugmentedInference(ts);

				double[] dpsi = VectorOperations.add(psi(ts.input, ts.output), -1, psi(ts.input, ystar));
				double[] ws = VectorOperations.mul(dpsi, 1/(lambda*l.size()));
				double ls = delta(ts.output,ystar)/l.size();

				double[] diff = VectorOperations.add(wall[i], -1, ws);
				double denom = VectorOperations.dot(diff, diff);
				double gamma = 0;
				if(denom != 0) {
					gamma = (lambda * VectorOperations.dot(diff, w) - lall[i] + ls) / (lambda * denom);
				}

				// update wi
				double[] wik = wall[i].clone();
				VectorOp.add(wall[i], ws, 1-gamma, gamma);
				lall[i] = (1-gamma)*lall[i] + gamma*ls;

				//System.out.println("k \t" + Arrays.toString(wik));
				//System.out.println("k+1 \t" + Arrays.toString(wall[i]));

				// update w
				for(int j=0; j<w.length; j++) {
					w[j] += wall[i][j] - wik[j];
				}

				if(verbose >= 1) {
					System.out.println("epochs " + (k+1) + "/" + maxIter + "\t gamma= " + gamma + "\tl^(i)= " + lall[i] + "\ti= " + i);
				}
			}
			System.out.print(".");
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
		System.out.println("Learning: Block-Coordinate Primal-Dual Frank-Wolfe - maxIter= " + maxIter);
	}

	@Override
	public String toString() {
		String s = "ssvm_FrankWolfeBlockCoordinate_lambda_" + lambda + "_maxIter_" + maxIter;
		return s;
	}
}
